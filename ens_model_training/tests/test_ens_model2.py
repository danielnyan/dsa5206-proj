"""ENS model-2 contracts on CPU: synthetic images/ZIPs only, no dataset or weight downloads."""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import itertools
import json
import math
from pathlib import Path
import zipfile

import numpy as np
from PIL import Image
import pytest

from ens_model_training import ens_cache as e
from ens_model_training import ens_train as t
from modern_pca import full_cache as f
from modern_pca import preprocessing_cache as c
from modern_pca import reimplementation as r



class Identity:
    def transform(self, pixels):
        return pixels.copy()


IDENTITY = (Identity(), Identity())


def jpeg(seed, size=(612, 610)):
    stream = io.BytesIO()
    Image.fromarray(np.random.default_rng(seed).integers(0, 256, (*size[::-1], 3), dtype=np.uint8)).save(stream, 'JPEG')
    return stream.getvalue()


def row(i, label, split, payload):
    folder, stem = ('norm', 'norm') if label == 'benign' else ('tu', 'tum')
    relative = f'{label}/{folder}/{stem}.{i}.jpg'
    return dict(sample_id='VAL1:' + relative, relative_path=relative, cohort='VAL1', class_name=label,
                ground_truth_code=r.CLASSES[label], file_sha256=hashlib.sha256(payload).hexdigest(), split=split)


# ---- model-2 input forming ----

def test_model2_input_is_model1_pipeline_with_only_the_resize_changed(tmp_path):
    payload = jpeg(1); path = tmp_path / 'x.jpg'; path.write_bytes(payload)
    item = row(0, 'tumour', 'train', payload)
    at350 = e.model2_pixels(item, payload, IDENTITY, size=350)
    np.testing.assert_array_equal(at350, c.normalized_uint8(item, path, IDENTITY))
    np.testing.assert_array_equal(e.to_float(at350), r.preprocess_legacy(item, path, IDENTITY))
    np.testing.assert_array_equal(r.preprocess_legacy(item, e.Payload(payload), IDENTITY), r.preprocess_legacy(item, path, IDENTITY))
    at500 = e.model2_pixels(item, payload, IDENTITY)
    expected = np.array(Image.open(io.BytesIO(payload)).convert('RGB').resize((500, 500), Image.Resampling.LANCZOS))
    assert at500.shape == e.SHAPE and at500.dtype == np.uint8
    np.testing.assert_array_equal(at500, expected)  # Whole native patch, not a crop.
    with pytest.raises(ValueError, match='checksum'):
        e.model2_pixels(dict(item, file_sha256='0' * 64), payload, IDENTITY)


def test_magnification_and_model2_parameter_contract():
    assert t.nasnet_grid(350) == 11 and t.nasnet_grid(500) == 16  # Paper Extended Data Fig. 2: [11, 11, 4032].
    assert t.NASNET_BACKBONE == 84916818 and t.PARAMETERS == 349158740
    assert t.NASNET_BACKBONE + t.head_parameters(350) == r.PARAMETERS
    assert round(40 * e.SIZE / 600, 1) == 33.3 and round(40 * 350 / 600, 1) == 23.3


def test_part_plan_covers_frozen_split_within_output_cap():
    rows = [dict(split=s) for s, n in f.COUNTS.items() for _ in range(n)]
    parts = e.plan_parts(rows)
    shards = [s for part in parts for s in part]
    assert len(parts) == 8 and len(shards) == 570 and sum(len(b) for _, b in shards) == 145819
    assert all(len({x['split'] for x in batch}) == 1 for _, batch in shards)
    assert max(sum(e.shard_bytes(len(b)) for _, b in part) for part in parts) < 13.9e9


# ---- ZIP source and MD5-verified download ----

@pytest.fixture
def zips(tmp_path, monkeypatch):
    rows, archives = [], {}
    for label in r.CLASSES:
        name = f'val_dataset_1_{"norm" if label == "benign" else "tu"}.zip'
        path = tmp_path / 'zenodo' / name; path.parent.mkdir(exist_ok=True)
        with zipfile.ZipFile(path, 'w') as archive:
            for i in range(5):
                payload = jpeg(len(rows), (61 + i % 2, 62))
                item = row(i, label, 'train' if i < 3 else 'internal_validation', payload)
                archive.writestr(item['relative_path'].split('/', 1)[1], payload); rows.append(item)
        archives[label] = (name, e.md5_file(path), path.stat().st_size)
    monkeypatch.setattr(e, 'ARCHIVES', archives)
    monkeypatch.setattr(e, 'ZENODO', (tmp_path / 'zenodo').as_uri() + '/{}')
    order = {'train': 0, 'internal_validation': 1}
    rows.sort(key=lambda x: (order[x['split']], x['sample_id']))
    return rows, tmp_path


def test_zip_source_reads_authenticated_val1_members_only(zips):
    rows, tmp = zips
    source = e.ZipSource({label: tmp / 'zenodo' / name for label, (name, _, _) in e.ARCHIVES.items()})
    source.check(rows)
    assert hashlib.sha256(source.read(rows[0])).hexdigest() == rows[0]['file_sha256']
    with pytest.raises(ValueError, match='VAL2'):
        source.member(dict(rows[0], cohort='VAL2'))
    with pytest.raises(ValueError, match='checksum'):
        source.read(dict(rows[0], file_sha256='0' * 64))
    with pytest.raises(ValueError, match='absent'):
        source.check([dict(rows[0], relative_path='benign/norm/missing.jpg', sample_id='VAL1:benign/norm/missing.jpg')])


def test_download_verifies_md5_restarts_stale_partial_and_reuses_found_copies(zips):
    _, tmp = zips
    name = e.ARCHIVES['benign'][0]
    (tmp / 'scratch').mkdir(); (tmp / 'scratch' / (name + '.partial')).write_bytes(b'stale bytes')
    paths = e.download(tmp / 'scratch')
    assert all(e.md5_file(paths[k]) == e.ARCHIVES[k][1] for k in e.ARCHIVES)
    assert not list((tmp / 'scratch').glob('*.partial'))
    assert e.download(tmp / 'other', search=[tmp / 'zenodo'])['benign'] == tmp / 'zenodo' / name
    e.ARCHIVES['tumour'] = (e.ARCHIVES['tumour'][0], '0' * 32, 0)
    with pytest.raises(ValueError, match='MD5 mismatch'):
        e.download(tmp / 'third')
    assert not (tmp / 'third' / e.ARCHIVES['tumour'][0]).exists()


# ---- committed parts and the multi-part reader ----

@pytest.fixture
def cache(zips, monkeypatch):
    rows, tmp = zips
    monkeypatch.setattr(e, 'SHARD_SIZE', 2); monkeypatch.setattr(e, 'PART_SHARDS', 2)
    monkeypatch.setattr(f, 'frozen_rows', lambda: rows)
    monkeypatch.setattr(r, 'normalizers', lambda *args: IDENTITY)
    monkeypatch.setattr(e, 'ProcessPoolExecutor', lambda **kw: ThreadPoolExecutor(1, initializer=kw['initializer'], initargs=kw['initargs']))
    parts = len(e.plan_parts(rows))
    metas = [e.build_part(k, tmp / 'scratch', tmp / f'part{k}', workers=1, search=[tmp / 'zenodo']) for k in range(parts)]
    return rows, tmp, metas


def test_parts_rebuild_exact_tensors_in_frozen_order(cache):
    rows, tmp, metas = cache
    assert len(metas) == 3 and len({m['fingerprint'] for m in metas}) == 1
    reader = e.Model2Cache([tmp], rows)
    assert [x['sample_id'] for x in reader.items] == [x['sample_id'] for x in rows]
    source = e.ZipSource({label: tmp / 'zenodo' / name for label, (name, _, _) in e.ARCHIVES.items()})
    for i, item in enumerate(rows):
        np.testing.assert_array_equal(reader.get(i), e.model2_pixels(item, source.read(item), IDENTITY))
    assert reader.indices('internal_validation') == [i for i, x in enumerate(rows) if x['split'] == 'internal_validation']


@pytest.mark.parametrize('fault', ['missing', 'duplicate', 'context', 'tensor', 'truncate', 'incomplete'])
def test_part_faults_rejected(cache, fault):
    import shutil
    rows, tmp, _ = cache
    if fault == 'missing':
        (tmp / 'part1' / 'ens_part.json').unlink()
    if fault == 'duplicate':
        shutil.copytree(tmp / 'part1', tmp / 'part1copy')
    if fault == 'context':
        meta = f.read(tmp / 'part2' / 'ens_part.json'); meta['context']['software']['Pillow'] = 'other'
        meta['fingerprint'] = f.digest(meta['context']); f.atomic_json(tmp / 'part2' / 'ens_part.json', meta)
    if fault == 'incomplete':
        meta = f.read(tmp / 'part0' / 'ens_part.json'); meta['status'] = 'RUNNING'; f.atomic_json(tmp / 'part0' / 'ens_part.json', meta)
    shard = tmp / 'part0' / 'train_00000.npy'
    if fault == 'truncate':
        with shard.open('r+b') as stream: stream.truncate(1000)
    if fault == 'tensor':
        data = np.load(shard, mmap_mode='r+'); data[0, 0, 0, 0] ^= 1; data.flush(); del data
        with pytest.raises(ValueError, match='checksum'):
            e.Model2Cache([tmp], rows).get(0)
        return
    with pytest.raises(ValueError):
        e.Model2Cache([tmp], rows)


def test_nonempty_part_output_and_val2_rows_refused(cache):
    rows, tmp, metas = cache
    with pytest.raises(ValueError, match='nonempty'):
        e.build_part(0, tmp / 'scratch', tmp / 'part0', workers=1, search=[tmp / 'zenodo'])
    shard = metas[0]['shards'][0]
    with pytest.raises(ValueError, match='VAL2'):
        e.check_items(shard['items'], [dict(x, cohort='VAL2') for x in rows[:2]], shard['name'])


# ---- schedule, ordering and the ENS rule (pure logic) ----

def test_stage_schedule_matches_model1():
    assert [t.stage_of(k) for k in (1, 7)] == [('normal_conv_1_17', 1e-5)] * 2
    assert [t.stage_of(k)[0] for k in range(8, 15)] == [f'normal_conv_1_{i}' for i in (14, 11, 9, 7, 5, 3, 1)]
    assert t.stage_of(8)[1] == 1e-6 and t.EPOCHS == 14
    for bad in (0, 15):
        with pytest.raises(ValueError):
            t.stage_of(bad)


class FakeReader:
    """In-memory stand-in for Model2Cache using the real ordering code."""
    indices = e.Model2Cache.indices

    def __init__(self, train=230, validation=210, shape=(4, 4, 3)):
        rng = np.random.default_rng(7)
        self.items = [dict(sample_id=f'VAL1:{"benign" if i % 3 else "tumour"}/x/{i}.jpg', ground_truth_code=int(i % 3 == 0),
                           split='train' if i < train else 'internal_validation') for i in range(train + validation)]
        self.pixels = rng.integers(0, 256, (train + validation, *shape), dtype=np.uint8)

    def get(self, index):
        return self.pixels[int(index)].copy()


def test_epoch_order_is_model1_order_and_differs_by_epoch():
    reader = FakeReader()
    train = [i for i, x in enumerate(reader.items) if x['split'] == 'train']
    for epoch in (1, 2):
        assert reader.indices('train', epoch) == sorted(train, key=lambda i: r.rank_id(reader.items[i]['sample_id'], f'VAL1-epoch-v1|42|{epoch}|'))
    assert reader.indices('train', 1) != reader.indices('train', 2) and sorted(reader.indices('train', 3)) == train


def test_ens_combination_rule():
    p1 = [0.05, 0.2, 0.5, 0.8, 0.81, 0.19, 0.7]
    p2 = [0.9, 0.9, 0.1, 0.3, 0.0, 1.0, 0.7]
    final, triggered = t.ens_combine(p1, p2)
    assert triggered.tolist() == [False, True, True, True, False, False, True]  # Grey zone [0.2, 0.8] inclusive.
    assert final.tolist() == [0.05, 0.9, 0.1, 0.3, 0.81, 0.19, 0.7]
    # Paper Fig. 2d example: model 1 tumour 0.28 (benign call) -> model 2 tumour 0.70 -> tumour.
    final, _ = t.ens_combine([0.28], [0.70])
    assert (final > 0.5).tolist() == [True]
    for a, b in (([0.5], [0.5, 0.5]), ([1.1], [0.5]), ([np.nan], [0.5]), ([0.5], [-0.1])):
        with pytest.raises(ValueError):
            t.ens_combine(a, b)


# ---- TensorFlow: loader, accumulation flush and exact multi-session resume ----

def tiny_model(seed=3):
    import tensorflow as tf
    model = tf.keras.Sequential([tf.keras.Input((4, 4, 3)), tf.keras.layers.Flatten(), tf.keras.layers.Dense(2, activation='softmax')])
    rng = np.random.default_rng(seed)
    model.set_weights([rng.normal(0, 0.1, w.shape).astype(np.float32) for w in model.get_weights()])
    return model


CONTRACT = dict(purpose=t.PURPOSE, fixture=True, schedule=r.SCHEDULE)


def run(output, resume=None, stop_at=math.inf, batch_size=4, model=None):
    model = model or tiny_model()
    state = t.train(model, FakeReader(), output, CONTRACT, batch_size=batch_size, resume=resume, stop_at=stop_at,
                    every=math.inf, clock=itertools.count().__next__, configure=lambda *_: None)
    return model, state


def test_batches_order_scaling_flips_and_labels():
    pytest.importorskip('tensorflow')
    reader = FakeReader()
    order = reader.indices('train', 2)[:10]
    seen = []
    for images, labels, chunk in t.batches(reader, order, 4, epoch=2):
        for image, label, i in zip(images.numpy(), labels, chunk):
            np.testing.assert_array_equal(image, np.asarray(r.augment(e.to_float(reader.get(i)), reader.items[i]['sample_id'], 2)))
            assert label == reader.items[i]['ground_truth_code']; seen.append(i)
    assert seen == order
    for images, _, chunk in t.batches(reader, reader.indices('internal_validation')[:3], 4):
        np.testing.assert_array_equal(images.numpy(), np.stack([e.to_float(reader.get(i)) for i in chunk]))


def test_interrupted_sessions_resume_bit_exactly(tmp_path):
    tf = pytest.importorskip('tensorflow')
    tf.config.experimental.enable_op_determinism()
    reference, done = run(tmp_path / 'straight')
    assert done['epoch'] == 15 and len(done['history']) == 14
    assert [h['updates'] for h in done['history']] == [3] * 14  # 230 = 100 + 100 + final partial 30, never dropped.
    assert len(list((tmp_path / 'straight').glob('checkpoint_e*'))) == 1  # Rotation keeps the output small.
    # 6 boundaries per epoch (3 updates, validation at 100 and 200, epoch end); stopping every 7th lands
    # after an update, at a training end, mid-validation, and on the epoch 7 -> 8 stage change.
    output = tmp_path / 'sessions'; resume, stops = None, []
    while not stops or stops[-1][0] <= t.EPOCHS:
        # Later sessions start from deliberately wrong weights: the restore must overwrite them.
        model, state = run(output, resume=resume, stop_at=7, model=tiny_model(seed=3 if resume is None else 99))
        resume = t.latest_checkpoint([output]); stops.append(t.progress(state))
    assert len(stops) == 12
    assert {(2, False, 100), (4, False, 230), (5, True, 100), (6, True, 200), (8, False, 0)} <= set(stops)
    for a, b in zip(reference.get_weights(), model.get_weights()):
        np.testing.assert_array_equal(a, b)
    assert json.dumps(state['history'], sort_keys=True) == json.dumps(done['history'], sort_keys=True)
    assert state['final_scores'] == done['final_scores']


def test_physical_batch_only_changes_float_summation(tmp_path):
    pytest.importorskip('tensorflow')
    a, _ = run(tmp_path / 'b4', batch_size=4)
    b, _ = run(tmp_path / 'b2', batch_size=2)
    for x, y in zip(a.get_weights(), b.get_weights()):
        np.testing.assert_allclose(x, y, rtol=1e-5, atol=1e-6)


def test_checkpoint_guards(tmp_path):
    tf = pytest.importorskip('tensorflow')
    model = tiny_model(); optimizer = r.adam(1e-5); accumulator = r.Accumulator(model, optimizer)
    accumulator.add(tf.ones((1, 4, 4, 3)), np.zeros(1, np.int64))
    with pytest.raises(ValueError, match='incomplete'):
        t.save(model, optimizer, accumulator, tmp_path, t.initial_state(), CONTRACT)
    accumulator.flush()
    (tmp_path / 'a' / 'checkpoint_e01_train_000000').mkdir(parents=True)  # Remnant of a session killed mid-write.
    early = t.save(model, optimizer, accumulator, tmp_path / 'a', t.initial_state(), CONTRACT)
    assert (early / t.CHECKPOINT).exists()
    later = t.save(model, optimizer, accumulator, tmp_path / 'b', dict(t.initial_state(), phase='validation'), CONTRACT)
    assert t.latest_checkpoint([tmp_path / 'a', tmp_path / 'b']) == later and early.exists()
    with pytest.raises(ValueError, match="mismatch in \\['tensorflow'\\]"):
        t.restore(tiny_model(), r.adam(1e-5), later, dict(CONTRACT, tensorflow='other'))
    (later / 'state.index').write_bytes(b'corrupt')
    with pytest.raises(Exception, match='SHA-256'):
        t.restore(tiny_model(), r.adam(1e-5), later, CONTRACT)


def test_export_binds_model_predictions_and_metadata(tmp_path):
    pytest.importorskip('tensorflow')
    model, state = run(tmp_path)
    meta = t.export(model, FakeReader(), tmp_path, state, {})
    assert meta['completed_epochs'] == 14 and meta['model_sha256'] == r.sha256_file(tmp_path / t.MODEL)
    rows = [x for x in FakeReader().items if x['split'] == 'internal_validation']
    _, scores = t.read_model2(tmp_path, [dict(sample_id=x['sample_id'], ground_truth_code=x['ground_truth_code']) for x in rows])
    assert scores == state['final_scores']
    with pytest.raises(ValueError, match='split'):
        t.read_model2(tmp_path, [dict(sample_id=x['sample_id'], ground_truth_code=x['ground_truth_code']) for x in rows[1:]])

