"""Synthetic cache contract tests: no external images or model downloads."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest

from modern_pca import full_cache as f
from modern_pca import preprocessing_cache as c
from modern_pca import reimplementation as r


class Identity:
    def transform(self, pixels):
        return pixels.copy()


def test_contract_fingerprint_survives_json_roundtrip():
    contract = {'shape': c.SHAPE, 'settings': {'dtype': 'uint8'}}
    assert f.digest(contract) == f.digest(json.loads(json.dumps(contract)))


@pytest.fixture
def full(tmp_path, monkeypatch):
    source = tmp_path / 'source'; rows = []
    for split in f.COUNTS:
        for label, code in r.CLASSES.items():
            path = source / f"val_dataset_1_{'norm' if code == 0 else 'tu'}" / f'{split}.jpg'
            path.parent.mkdir(parents=True, exist_ok=True)
            Image.fromarray(np.random.default_rng(len(rows)).integers(0, 256, (32, 31, 3), dtype=np.uint8)).save(path)
            relative = f'{label}/{split}.jpg'
            rows.append(dict(sample_id='VAL1:' + relative, relative_path=relative, cohort='VAL1', class_name=label,
                             ground_truth_code=code, file_sha256=r.sha256_file(path), split=split, width=31, height=32))
    context = dict(row_mapping_sha256=f.digest(rows), shard_size=2, expected_storage=c.storage_requirement((2, 2), 2))
    monkeypatch.setattr(f, 'preflight', lambda *args: (rows, context, {}))
    monkeypatch.setattr(f, 'worker_init', lambda *args: None)
    monkeypatch.setattr(f, '_NORMALIZERS', (Identity(), Identity()))
    monkeypatch.setattr(f, 'ProcessPoolExecutor', lambda **kwargs: ThreadPoolExecutor(max_workers=1))
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in source.rglob('*.jpg')}
    root = tmp_path / 'cache'
    f.generate(source, root, workers=1, shard_size=2)
    yield rows, source, root
    assert all(p.read_bytes() == value and p.stat().st_mtime_ns == stamp for p, (value, stamp) in before.items())


def test_full_contract_lookup_and_source_immutability(full):
    rows, source, root = full
    reader = f.FullCacheReader(root, rows, verify_all=True)
    assert reader.meta['samples'] == 4 and reader.meta['counts'] == {'train': 2, 'internal_validation': 2}
    for i, row in enumerate(rows):
        expected = c.normalized_uint8(row, r.image_path(row, source), (Identity(), Identity()))
        np.testing.assert_array_equal(reader.get(reader.by_id[row['sample_id']]), expected)
    assert reader.indices('train') == [0, 1] and reader.indices('internal_validation') == [2, 3]
    assert reader.indices('train', 3) == reader.indices('train', 3)


@pytest.mark.parametrize('fault', ['duplicate', 'split', 'count', 'label', 'checksum', 'offset', 'orphan'])
def test_catalog_corruption_rejected(full, fault):
    rows, _, root = full; path = root / 'cache.json'; value = f.read(path)
    if fault == 'duplicate': value['items'][1] = value['items'][0]
    if fault == 'split': value['items'][0]['split'] = 'internal_validation'
    if fault == 'count': value['samples'] -= 1
    if fault == 'label': value['items'][0]['class_name'] = 'gland'
    if fault == 'checksum': value['items'][0]['tensor_sha256'] = ''
    if fault == 'offset': value['items'][0]['index'] = 1
    if fault == 'orphan': value['items'].append(value['items'][0])
    f.atomic_json(path, value)
    with pytest.raises(ValueError): f.FullCacheReader(root, rows)


def test_shard_corruption_and_truncation(full):
    rows, _, root = full; path = root / 'train_00000.npy'
    data = np.load(path, mmap_mode='r+'); data[0, 0, 0, 0] ^= 1; data.flush(); del data
    reader = f.FullCacheReader(root, rows)
    with pytest.raises(ValueError, match='checksum'): reader.get(0)
    with pytest.raises(Exception): f.FullCacheReader(root, rows, verify_all=True)


def test_resume_does_not_overwrite_completed_shards(full):
    rows, source, root = full
    before = {p: (r.sha256_file(p), p.stat().st_mtime_ns) for p in root.glob('*.npy')}
    f.generate(source, root, workers=1, shard_size=2)
    assert all((r.sha256_file(p), p.stat().st_mtime_ns) == v for p, v in before.items())
    assert f.read(root / 'generation_sessions.json')[-1]['resumed_samples'] == 4


def test_interrupted_shard_preserved_and_rebuilt(full):
    rows, source, root = full
    # Simulate interruption between NPY rename and commit-sidecar write.
    (root / 'internal_validation_00000.npy.json').unlink()
    original = r.sha256_file(root / 'internal_validation_00000.npy')
    history = f.read(root / 'generation_sessions.json')
    history[-1]['status'] = 'RUNNING'
    f.atomic_json(root / 'generation_sessions.json', history)
    f.generate(source, root, workers=1, shard_size=2)
    preserved = list((root / 'interrupted').iterdir())
    assert len(preserved) == 1 and r.sha256_file(preserved[0]) == original
    assert f.FullCacheReader(root, rows, verify_all=True).meta['samples'] == 4
    metadata = f.read(root / 'cache.json')
    assert metadata['retries'] == 1 and metadata['interrupted_sessions'] == 1
    assert metadata['generation_timing_is_lower_bound']


def test_orphan_incomplete_and_stale_context_rejected(full):
    rows, _, root = full
    (root / 'orphan.npy').write_bytes(b'not a shard')
    with pytest.raises(ValueError, match='Orphan'): f.FullCacheReader(root, rows)
    (root / 'orphan.npy').unlink()
    (root / 'interrupted.partial').write_bytes(b'partial')
    with pytest.raises(ValueError, match='Incomplete'): f.FullCacheReader(root, rows)
    (root / 'interrupted.partial').unlink()
    context = f.read(root / 'context.json'); context['context']['shard_size'] = 7
    f.atomic_json(root / 'context.json', context)
    with pytest.raises(ValueError, match='fingerprint'): f.FullCacheReader(root, rows)


def test_deterministic_audit_covers_splits_classes_and_shards(full):
    rows, _, root = full; items = f.read(root / 'cache.json')['items']
    a, b = f.audit_indices(rows, items, 3)
    assert (a, b) == f.audit_indices(rows, items, 3)
    assert {rows[i]['split'] for i in a} == set(f.COUNTS)
    assert {rows[i]['class_name'] for i in a} == set(r.CLASSES)
    assert {items[i]['shard'] for i in a} == {x['shard'] for x in items}
    assert len(b) == len(set(b)) == 3


def test_val2_and_split_crossover_rejected(full):
    rows, _, root = full; items = f.read(root / 'cache.json')['items'][:2]
    altered = [dict(rows[0], cohort='VAL2'), rows[1]]
    with pytest.raises(ValueError): f.validate_items(items, altered, 'train_00000.npy')
    with pytest.raises(ValueError): r.image_path(altered[0], root)


def test_cache_dataset_order_scaling_and_training_only_flips(full):
    import tensorflow as tf
    rows, _, root = full; reader = f.FullCacheReader(root, rows)
    for split in f.COUNTS:
        indices = reader.indices(split, 2 if split == 'train' else None)
        observed = []
        for images, labels, ids in c.dataset(reader, indices, batch_size=1, epoch=2, augment=split == 'train'):
            i = int(ids.numpy()[0]); expected = c.to_float(reader.get(i))
            if split == 'train': expected = np.asarray(r.augment(expected, rows[i]['sample_id'], 2))
            np.testing.assert_array_equal(images.numpy()[0], expected)
            assert int(labels.numpy()[0]) == rows[i]['ground_truth_code']
            observed.append(i)
        assert observed == indices


def test_trainer_batch_adapter_and_resume_boundary(full):
    from modern_pca import cached_training as t
    rows, _, root = full; reader = f.FullCacheReader(root, rows)
    for split in f.COUNTS:
        actual = []
        for images, labels, ids in t.batches(reader, split, 1):
            for image, label, i in zip(images.numpy(), labels.numpy(), ids.numpy()):
                expected = c.to_float(reader.get(i))
                if split == 'train': expected = np.asarray(r.augment(expected, rows[i]['sample_id'], 1))
                np.testing.assert_array_equal(image, expected)
                assert label == rows[i]['ground_truth_code']
                actual.append(int(i))
        assert actual == reader.indices(split, 1 if split == 'train' else None)
    with pytest.raises(ValueError, match='boundary'):
        t.batches(reader, 'train', 1, start=1)


def test_trainer_partial_logical_batch_and_checkpoint_restore(tmp_path):
    import tensorflow as tf
    from modern_pca import cached_training as t
    model = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(2, activation='softmax')])
    optimizer = r.adam(1e-5); acc = r.Accumulator(model, optimizer)
    data = [(tf.ones((min(4, 155 - i), 2)), tf.zeros((min(4, 155 - i),), tf.int64),
             tf.range(i, min(i + 4, 155))) for i in range(0, 155, 4)]
    result = t.train_batches(acc, data)
    assert result['samples'] == 155 and result['updates'] == 2 and acc.count == 0
    assert int(optimizer.iterations.numpy()) == 2
    expected = model(tf.ones((2, 2))).numpy()
    state = dict(purpose='fixture', epoch=1, next_offset=100)
    root = tmp_path / 'checkpoint'
    t.save_checkpoint(model, optimizer, acc, root, state, 'a' * 64)
    model.trainable_variables[-1].assign_add(tf.ones_like(model.trainable_variables[-1]))
    optimizer.iterations.assign_add(7)
    restored = t.restore_checkpoint(model, optimizer, root, 'a' * 64, 'fixture')
    assert restored == state and int(optimizer.iterations.numpy()) == 2
    np.testing.assert_array_equal(expected, model(tf.ones((2, 2))).numpy())
    with pytest.raises(ValueError, match='purpose'):
        t.restore_checkpoint(model, optimizer, root, 'a' * 64, 'production')
    with pytest.raises(ValueError, match='mismatch'):
        t.restore_checkpoint(model, optimizer, root, 'b' * 64, 'fixture')
    acc.add(tf.ones((1, 2)), tf.zeros((1,), tf.int64))
    with pytest.raises(ValueError, match='incomplete'):
        t.save_checkpoint(model, optimizer, acc, tmp_path / 'bad', state, 'a' * 64)


def test_cached_production_requires_explicit_mode_and_batch_four():
    from modern_pca.train_reimplementation import parse_args
    with pytest.raises(SystemExit):
        parse_args(['--train-production', '--cache', 'cache', '--output', 'run'])
    with pytest.raises(SystemExit):
        parse_args(['--freeze-split', '--cache', 'cache', '--batch-size', '4'])
    args = parse_args(['--train-production', '--cache', 'cache', '--output', 'run', '--batch-size', '4'])
    assert args.extracted is None and args.cache == Path('cache')


@pytest.mark.parametrize('fault', ['truncate', 'dtype'])
def test_invalid_shard_structure_rejected(full, fault):
    rows, _, root = full; path = root / 'train_00000.npy'
    if fault == 'truncate':
        with path.open('r+b') as stream: stream.truncate(256)
    else:
        np.save(path, np.zeros((2, *c.SHAPE), np.float32), allow_pickle=False)
    with pytest.raises((ValueError, OSError)):
        f.FullCacheReader(root, rows)


def test_full_cache_forecast_accounts_for_input_and_resume_checkpoints():
    from tools.report_stage2g import forecast
    previous = dict(stages=[dict(boundary='fixture', epochs=14, train_images_per_second=100., validation_images_per_second=200.)],
                    checkpoint_io_seconds=10., startup_compilation_seconds=5.)
    loader = dict(splits={'train': dict(images_per_second=50.), 'internal_validation': dict(images_per_second=100.)})
    value = forecast(previous, loader, 2.)
    assert value['epochs_14_seconds'] == pytest.approx(14 * (116655 / 50 + 29164 / 100) + 15 + 28)
    assert value['epoch_resume_checkpoint_seconds'] == 28
    assert value['conservative_serial_seconds'] > value['epochs_14_seconds']


def test_portable_checkpoint_export_path(tmp_path):
    import tensorflow as tf
    from modern_pca import cached_training as t
    model = tf.keras.Sequential([tf.keras.Input((2,)), tf.keras.layers.Dense(2, activation='softmax')])
    result = t.portable_roundtrip(model, tmp_path / 'disposable.keras', tf.ones((1, 2)))
    assert result['status'] == 'PASS' and result['weights_exact'] and result['prediction_exact']
    with pytest.raises(ValueError, match='overwrite'):
        t.portable_roundtrip(model, tmp_path / 'disposable.keras', tf.ones((1, 2)))
