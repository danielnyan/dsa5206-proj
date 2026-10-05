"""Stage 2H recovery tests use synthetic CPU models; no production updates/data."""
from copy import deepcopy
import json
import logging
from pathlib import Path

import numpy as np
import pytest

from modern_pca import cached_training as t, full_cache as f, reimplementation as r
from modern_pca import train_reimplementation as cli
from modern_pca.evaluate_paper import EvaluationError


def contract():
    return dict(provenance=dict(environment=dict(platform=dict(ram_bytes=16511066112, cpu='x86_64'),
                                                software=dict(tensorflow='2.20.0')),
                               settings=dict(shape=(350, 350, 3), dtype='uint8'),
                               preprocessing_implementation_sha256='preprocessing',
                               source_VAL1_manifest_sha256='manifest', split_sha256={'train': 'split'},
                               stain_reference_sha256='stain'),
                backend_identity={'spams-bin': {'files_sha256': {'backend.so': 'hash'}}})


@pytest.mark.parametrize('ram', [16511066112, 16511041536])
def test_environment_exact_or_ram_only_accepted(ram, caplog):
    cached = json.loads(json.dumps(contract())); current = contract()
    current['provenance']['environment']['platform']['ram_bytes'] = ram
    snapshots = deepcopy((cached, current))
    with caplog.at_level('INFO'):
        result = t.check_environment_compatibility(cached, current)
    assert result['status'] == 'PASS' and (cached, current) == snapshots
    assert result['informational_ram_bytes'] == {'cached': 16511066112, 'current': ram}
    if ram != 16511066112:
        assert '16511066112' in caplog.text and str(ram) in caplog.text


@pytest.mark.parametrize('path', [
    ('provenance', 'settings', 'shape'), ('provenance', 'settings', 'dtype'),
    ('provenance', 'preprocessing_implementation_sha256'),
    ('provenance', 'environment', 'software', 'tensorflow'),
    ('backend_identity', 'spams-bin', 'files_sha256', 'backend.so'),
    ('provenance', 'source_VAL1_manifest_sha256'), ('provenance', 'split_sha256', 'train'),
    ('provenance', 'stain_reference_sha256'), ('provenance', 'environment', 'platform', 'cpu'),
])
def test_every_other_contract_difference_rejected(path):
    cached = contract(); current = deepcopy(cached)
    current['provenance']['environment']['platform']['ram_bytes'] -= 24576
    node = current
    for key in path[:-1]: node = node[key]
    node[path[-1]] = 'changed'
    with pytest.raises(ValueError, match='environment mismatch'):
        t.check_environment_compatibility(cached, current)


def test_missing_ram_field_rejected():
    cached = contract(); current = contract()
    del current['provenance']['environment']['platform']['ram_bytes']
    with pytest.raises(ValueError, match='missing RAM'):
        t.check_environment_compatibility(cached, current)


def history():
    from modern_pca.evaluate_paper import calculate_binary_metrics
    labels = [0] * r.VALIDATION_COUNTS['benign'] + [1] * r.VALIDATION_COUNTS['tumour']
    metrics = calculate_binary_metrics(labels, np.asarray(labels) * .8 + .1)
    boundaries = [name for name, epochs, _ in r.SCHEDULE for _ in range(epochs)]
    return [dict(epoch=i, boundary=b, loss=.25, internal_validation=deepcopy(metrics))
            for i, b in enumerate(boundaries, 1)]


@pytest.fixture
def recovery(tmp_path, monkeypatch):
    import tensorflow as tf
    def build():
        return tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(
            2, activation='softmax', kernel_initializer='zeros', bias_initializer='zeros')])
    model = build()
    # Distinct checkpoint weights and Adam slots, assigned without optimizer updates.
    model.weights[0].assign([[1., -1.], [2., -2.]])
    model.weights[1].assign([.5, -.5])
    optimizer = r.adam(1e-6); optimizer.build(model.trainable_variables)
    optimizer.iterations.assign(1167)
    for v in optimizer.variables[2:]: v.assign(tf.ones_like(v) * .125)
    state = dict(purpose='production', completed_epochs=14, boundary=r.SCHEDULE[-1][0],
                 rate=1e-6, history=history())
    cache = tmp_path / 'cache'; cache.mkdir(); (cache / 'cache.json').write_text('{}')
    cache_sha = r.sha256_file(cache / 'cache.json')
    resume = tmp_path / 'epoch_14_resume'
    from types import SimpleNamespace
    t.save_checkpoint(model, optimizer, SimpleNamespace(count=0), resume, state, cache_sha)
    expected = r.weight_hashes(model.weights)
    monkeypatch.setattr(t, 'checked_reader', lambda root: object())
    monkeypatch.setattr(r, 'gpu_setup', lambda: {})
    monkeypatch.setattr(r, 'build_model', lambda: (build(), {}))
    stages = []
    monkeypatch.setattr(r, 'configure_stage', lambda model, boundary: stages.append(boundary))
    monkeypatch.setattr(r, 'metadata', lambda *args: {})
    monkeypatch.setattr(r, 'freeze_split', lambda *args: ([], [], {}))
    def forbidden(*args, **kwargs):
        pytest.fail('Training/validation/optimizer update forbidden during finalization')
    monkeypatch.setattr(t, 'batches', forbidden)
    monkeypatch.setattr(t, 'train_batches', forbidden)
    monkeypatch.setattr(t, 'validation', forbidden)
    monkeypatch.setattr(r, 'Accumulator', forbidden)
    monkeypatch.setattr(tf.keras.optimizers.Adam, 'apply_gradients', forbidden)
    output = tmp_path / 'finalized'
    argv = ['--train-production', '--cache', str(cache), '--batch-size', '4',
            '--resume-cache-checkpoint', str(resume), '--output', str(output)]
    logger = logging.getLogger(); original_handlers = list(logger.handlers); level = logger.level
    try:
        yield dict(argv=argv, output=output, resume=resume, expected=expected, stages=stages,
                   initial=r.weight_hashes(build().weights), state=state)
    finally:
        for handler in list(logger.handlers):
            if handler not in original_handlers:
                logger.removeHandler(handler); handler.close()
        logger.setLevel(level)


def test_epoch14_cli_exports_strictly_restored_weights_without_updates_or_iterators(recovery):
    import tensorflow as tf
    x = recovery
    checkpoint_before = (x['resume'] / 'checkpoint.json').read_bytes()
    assert cli.main(x['argv']) == 0
    exported = tf.keras.models.load_model(x['output'] / 'epoch14.keras', compile=False)
    assert r.weight_hashes(exported.weights) == x['expected'] != x['initial']
    meta = f.read(x['output'] / 'metadata.json')
    assert meta['completed_epochs'] == 14 and meta['finalization_only']
    assert meta['optimizer_iterations'] == 1167
    assert f.read(x['output'] / 'history.json')['epochs'] == x['state']['history']
    assert x['stages'] == [r.SCHEDULE[-1][0]]
    assert (x['resume'] / 'checkpoint.json').read_bytes() == checkpoint_before
    with pytest.raises(FileExistsError): cli.main(x['argv'])


@pytest.mark.parametrize('fault', ['purpose', 'boundary', 'rate', 'history_length', 'epoch_order',
                                 'history_boundary', 'loss', 'validation', 'counts', 'metric',
                                 'completed_type', 'completed_range', 'file_hash', 'tensor_hash',
                                 'optimizer_hash', 'iterations', 'contract'])
def test_malformed_or_diagnostic_checkpoint_never_finalizes(recovery, fault):
    x = recovery; path = x['resume'] / 'checkpoint.json'; value = f.read(path); state = value['state']
    if fault == 'purpose': state['purpose'] = 'stage2g_disposable'
    if fault == 'boundary': state['boundary'] = r.SCHEDULE[-2][0]
    if fault == 'rate': state['rate'] = 1e-5
    if fault == 'history_length': state['history'].pop()
    if fault == 'epoch_order': state['history'][5]['epoch'] = 5
    if fault == 'history_boundary': state['history'][13]['boundary'] = r.SCHEDULE[0][0]
    if fault == 'loss': state['history'][0]['loss'] = float('nan')
    if fault == 'validation': del state['history'][0]['internal_validation']
    if fault == 'counts': state['history'][0]['internal_validation']['TN'] -= 1
    if fault == 'metric': del state['history'][0]['internal_validation']['roc_auc']
    if fault == 'completed_type': state['completed_epochs'] = 14.0
    if fault == 'completed_range': state['completed_epochs'] = 15
    if fault == 'file_hash': value['files_sha256']['state.index'] = '0' * 64
    if fault == 'tensor_hash': value['model_tensor_sha256'][0] = '0' * 64
    if fault == 'optimizer_hash': value['optimizer_tensor_sha256'][0] = '0' * 64
    if fault == 'iterations': value['optimizer_iterations'] += 1
    if fault == 'contract': value['contract']['physical_batch'] = 1
    f.atomic_json(path, value)
    with pytest.raises((ValueError, EvaluationError)): cli.main(x['argv'])
    assert not (x['output'] / 'epoch14.keras').exists()


def test_epoch13_uses_normal_resume_path_without_executing_training(recovery, monkeypatch):
    x = recovery; path = x['resume'] / 'checkpoint.json'; value = f.read(path)
    value['state'].update(completed_epochs=13, boundary=r.SCHEDULE[-2][0], history=x['state']['history'][:13])
    f.atomic_json(path, value)
    restored = []
    real_restore = t.restore_checkpoint
    def restore(*args):
        state = real_restore(*args); restored.append(state['completed_epochs']); return state
    monkeypatch.setattr(t, 'restore_checkpoint', restore)
    monkeypatch.setattr(r, 'Accumulator', lambda *args: object())
    class TrainingPathReached(Exception): pass
    def batches(reader, split, epoch):
        assert split == 'train' and epoch == 14 and restored == [13]
        raise TrainingPathReached
    monkeypatch.setattr(t, 'batches', batches)
    with pytest.raises(TrainingPathReached): cli.main(x['argv'])
    assert not (x['output'] / 'epoch14.keras').exists()
    assert x['stages'] == [r.SCHEDULE[-1][0], r.SCHEDULE[-2][0], r.SCHEDULE[-1][0]]
