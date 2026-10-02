"""Authenticated, resumable VAL1-only uint8 cache. Never starts model training."""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict, defaultdict
from concurrent.futures import ProcessPoolExecutor, wait, FIRST_COMPLETED
import csv
import hashlib
import importlib.metadata
import json
import logging
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import threading
import time

import numpy as np

from . import preprocessing_cache as c
from . import reimplementation as r

VERSION = 'VAL1-uint8-npy-v1'
COUNTS = {'train': 116655, 'internal_validation': 29164}
LOG = logging.getLogger('full_cache')


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('w', encoding='utf-8', newline='\n') as stream:
        json.dump(value, stream, sort_keys=True, separators=(',', ':'))
        stream.write('\n'); stream.flush(); os.fsync(stream.fileno())
    temporary.replace(path)


def frozen_rows():
    """Read only authenticated VAL1 manifests; never discover/scan source folders."""
    source = r.read_manifest(Path('manifests/VAL1_manifest.csv'))
    sources = {row['sample_id']: row for row in source}
    result = []
    for split, name in [('train', 'VAL1_train.csv'), ('internal_validation', 'VAL1_internal_validation.csv')]:
        path = Path('manifests/VAL1_internal_v1') / name
        r.sha256_file(path, c.SPLIT_HASHES[name])
        with path.open(encoding='utf-8', newline='') as stream:
            rows = list(csv.DictReader(stream))
        for row in rows:
            for key in ('row_index', 'ground_truth_code', 'file_size_bytes', 'width', 'height'):
                row[key] = int(row[key])
            original = sources.get(row['sample_id'])
            if original is None or any(row[k] != original[k] for k in original if k != 'row_index'):
                raise ValueError('Frozen split/source mapping mismatch')
            row['split'] = split
        r.validate_rows(rows, 'VAL1', r.TRAIN_COUNTS if split == 'train' else r.VALIDATION_COUNTS)
        result.extend(rows)
    if len(result) != len(source) or len({x['sample_id'] for x in result}) != len(source):
        raise ValueError('Split crossover, missing or duplicate source sample')
    return result


def plan_shards(rows, shard_size):
    if shard_size < 1:
        raise ValueError('Invalid shard size')
    result = []
    for split in COUNTS:
        subset = [x for x in rows if x['split'] == split]
        for start in range(0, len(subset), shard_size):
            result.append((f'{split}_{start // shard_size:05d}.npy', subset[start:start + shard_size]))
    if sum(len(batch) for _, batch in result) != len(rows):
        raise ValueError('Unexpected split')
    return result


def identity(row):
    return {k: row[k] for k in ('sample_id', 'relative_path', 'class_name', 'ground_truth_code', 'cohort', 'split')}


def validate_items(items, rows, name):
    if len(items) != len(rows):
        raise ValueError('Manifest/cache count mismatch')
    ids, locations = set(), set()
    for index, (item, row) in enumerate(zip(items, rows)):
        if identity(item) != identity(row) or item['source_jpeg_sha256'] != row['file_sha256']:
            raise ValueError('Source/split mapping mismatch')
        if row['cohort'] != 'VAL1' or row['class_name'] not in r.CLASSES or row['ground_truth_code'] != r.CLASSES[row['class_name']]:
            raise ValueError('Unexpected cohort/class')
        if item['sample_id'] in ids or (item['shard'], item['index']) in locations:
            raise ValueError('Duplicate sample or manifest mapping')
        ids.add(item['sample_id']); locations.add((item['shard'], item['index']))
        if item['shard'] != name or item['index'] != index or item['shape'] != list(c.SHAPE) or item['dtype'] != 'uint8' or item['preprocessing_status'] != 'PASS':
            raise ValueError('Invalid cache tensor contract')
        if len(item['tensor_sha256']) != 64 or any(x not in '0123456789abcdef' for x in item['tensor_sha256']):
            raise ValueError('Invalid tensor checksum')


def verify_shard(root, name, rows, fingerprint, tensors=True):
    root = Path(root); path = root / name
    if Path(name).name != name or path.is_symlink() or path.with_suffix('.npy.json').is_symlink():
        raise ValueError('Unsafe shard path')
    sidecar = read(root / (name + '.json'))
    if sidecar['context_fingerprint'] != fingerprint or sidecar['status'] != 'COMPLETE':
        raise ValueError('Incomplete/stale shard')
    validate_items(sidecar['items'], rows, name)
    r.sha256_file(path, sidecar['file_sha256'])
    array = np.load(path, mmap_mode='r', allow_pickle=False)
    if array.dtype != np.uint8 or array.shape != (len(rows), *c.SHAPE) or array.flags.f_contiguous:
        raise ValueError('Invalid shard dtype/shape/layout')
    if path.stat().st_size != array.offset + array.nbytes:
        raise ValueError('Incomplete or trailing shard bytes')
    if tensors:
        for pixels, item in zip(array, sidecar['items']):
            if hashlib.sha256(pixels.tobytes()).hexdigest() != item['tensor_sha256']:
                raise ValueError('Tensor checksum mismatch')
    del array
    return sidecar


_NORMALIZERS = None


def worker_init(reference):
    global _NORMALIZERS
    _NORMALIZERS = r.normalizers(Path(reference))


def build_shard(root, extracted, name, rows, fingerprint):
    """One isolated worker owns one shard; commit sidecar LAST as transaction marker."""
    root = Path(root); path = root / name; sidecar = root / (name + '.json')
    if path.exists() or sidecar.exists():
        raise ValueError('Refusing to overwrite an existing shard')
    started = time.perf_counter(); cpu = time.process_time(); timings = defaultdict(float)
    temporary = root / (name + '.partial')
    array = np.lib.format.open_memmap(temporary, mode='w+', dtype='uint8', shape=(len(rows), *c.SHAPE))
    items = []
    for index, row in enumerate(rows):
        source = r.image_path(row, Path(extracted))
        before = source.stat()
        pixels = c.normalized_uint8(row, source, _NORMALIZERS, timings)
        array[index] = pixels
        item = c.item_metadata(row, name, index, pixels)
        item.update(split=row['split'], preprocessing_status='PASS', source_size_bytes=before.st_size,
                    source_mtime_ns=before.st_mtime_ns)
        items.append(item)
    array.flush(); del array
    with temporary.open('rb') as stream:
        os.fsync(stream.fileno())
    temporary.replace(path)
    result = dict(status='COMPLETE', context_fingerprint=fingerprint, items=items,
                  file_sha256=r.sha256_file(path), bytes=path.stat().st_size,
                  seconds=time.perf_counter() - started, cpu_seconds=time.process_time() - cpu,
                  components_seconds=dict(timings))
    atomic_json(sidecar, result)
    return dict(name=name, samples=len(rows), seconds=result['seconds'], cpu_seconds=result['cpu_seconds'])


def environment_contract():
    pins = {}
    for line in Path('requirements-reimplementation-wsl.txt').read_text().splitlines():
        if '==' in line and not line.startswith('#'):
            name, version = line.split('=='); name = name.split('[')[0]
            actual = importlib.metadata.version(name)
            if actual != version:
                raise ValueError(f'Pinned dependency mismatch: {name}: {actual} != {version}')
            pins[name] = actual
    if sys.version_info[:3] != (3, 11, 16):
        raise ValueError('Use the verified Python 3.11.16 WSL environment')
    provenance = c.cache_provenance()
    baseline = read('runs/stage2e_profile/cache/context.json')['context']['provenance']
    for key in ('source_VAL1_manifest_sha256', 'split_sha256', 'stain_reference_sha256', 'preprocessing_implementation_sha256', 'settings'):
        if digest(provenance[key]) != digest(baseline[key]):
            raise ValueError(f'Stage 2E preprocessing contract changed: {key}')
    backends = {}
    for name in ('spams-bin', 'opencv-python', 'staintools'):
        distribution = importlib.metadata.distribution(name)
        files = {str(p): r.sha256_file(Path(distribution.locate_file(p))) for p in distribution.files or []
                 if str(p).endswith(('.so', '.py')) and Path(distribution.locate_file(p)).is_file()}
        backends[name] = dict(version=distribution.version, files_sha256=files, fingerprint=digest(files))
    return dict(provenance=provenance, pinned_dependencies=pins, backend_identity=backends)


def preflight(extracted, output, shard_size=256):
    extracted, output = Path(extracted).resolve(), Path(output).resolve()
    r.ensure_output_separation(extracted, [output])
    # Also exclude the enclosing original-archive dataset directory.
    r.ensure_output_separation(extracted.parent, [output])
    if 'val2' in str(output).lower() or output.is_symlink():
        raise ValueError('Unsafe output location')
    rows = frozen_rows(); contract = environment_contract()
    r.sha256_file(r.REFERENCE, r.REFERENCE_SHA)
    protected = [Path('modern_pca/evaluate_paper.py'), Path('modern_pca/reimplementation.py'),
                 Path(c.__file__), Path('manifests/VAL1_manifest.csv'), r.REFERENCE,
                 *[Path('manifests/VAL1_internal_v1') / name for name in c.SPLIT_HASHES]]
    storage = c.storage_requirement(tuple(COUNTS.values()), shard_size)
    ancestor = output
    while not ancestor.exists():
        ancestor = ancestor.parent
    free = shutil.disk_usage(ancestor).free
    existing = sum(p.stat().st_size for p in output.glob('*.npy')) if output.exists() else 0
    required = max(70_000_000_000 - existing, storage['npy_bytes'] + 200_000_000 + 10_000_000_000 - existing)
    if free < required:
        raise ValueError(f'Insufficient free disk: {free} < {required}')
    value = dict(format_version=VERSION, expected_counts=COUNTS, expected_samples=len(rows),
                 shard_size=shard_size, order='train manifest order, then internal_validation manifest order; no shard crosses a split',
                 row_mapping_sha256=digest(rows), contract=contract,
                 preprocessing_settings=contract['provenance']['settings'],
                 protected_sha256={str(p): r.sha256_file(p) for p in protected},
                 expected_storage=storage)
    audit = dict(timestamp=r.utc_now(), git=r.capture_git(), extracted=str(extracted), output=str(output),
                 free_bytes=free, required_free_bytes=required, output_outside_source=True,
                 source_access='Exact authenticated VAL1 paths only, read_bytes/stat; no source writes or recursive discovery',
                 VAL2_accessed=False, implementation_sha256=r.sha256_file(Path(__file__)))
    return rows, value, audit


def quarantine_partial(root, path):
    """Preserve, rather than overwrite, uncommitted transaction remnants."""
    root = Path(root).resolve(); path = Path(path)
    if path.parent.resolve() != root or path.is_symlink():
        raise ValueError('Unsafe partial artifact')
    target = root / 'interrupted'
    target.mkdir(exist_ok=True)
    destination = target / (path.name + '.' + str(time.time_ns()))
    path.replace(destination)
    return str(destination)


def generate(extracted, output, workers=4, shard_size=256):
    rows, context, audit = preflight(extracted, output, shard_size)
    output = Path(output); output.mkdir(parents=True, exist_ok=True)
    fingerprint = digest(context)
    context_path = output / 'context.json'
    if context_path.exists():
        if read(context_path)['fingerprint'] != fingerprint:
            raise ValueError('Cache context mismatch; refuse resume')
    else:
        if any(output.iterdir()):
            raise ValueError('Unrecognized nonempty cache directory')
        atomic_json(context_path, dict(fingerprint=fingerprint, context=context, created_at_utc=r.utc_now()))
    atomic_json(output / 'preflight.json', audit)
    plan = plan_shards(rows, shard_size); names = {name for name, _ in plan}
    for path in output.glob('*.npy*'):
        base = path.name.split('.npy')[0] + '.npy'
        if base not in names:
            raise ValueError('Orphan/unexpected shard')
    pending = []; resumed = 0; preserved = []; retried_shards = 0
    for name, batch in plan:
        if (output / (name + '.json')).exists():
            verify_shard(output, name, batch, fingerprint)
            resumed += len(batch)
            if resumed % 4096 == 0:
                LOG.info('Resume checksum verification %d committed images', resumed)
        else:
            had_partial = False
            for suffix in ('', '.partial', '.json.tmp'):
                path = output / (name + suffix)
                if path.exists():
                    preserved.append(quarantine_partial(output, path))
                    had_partial = True
            retried_shards += int(had_partial)
            pending.append((name, batch))
    history_path = output / 'generation_sessions.json'
    sessions = read(history_path) if history_path.exists() else []
    for prior in sessions:
        if prior['status'] == 'RUNNING':
            prior.update(status='INTERRUPTED', timing_is_lower_bound=True,
                         interruption_note='Previous writer stopped without finalization; wall time is its last persisted monotonic checkpoint. Unrecorded tail and downtime are excluded.')
    session = dict(started_at_utc=r.utc_now(), workers=workers, resumed_samples=resumed,
                   completed_samples=0, status='RUNNING', failures=0, retries=retried_shards, preserved_partials=preserved,
                   wall_seconds=0., worker_cpu_seconds=0., process_cpu_percent_available=True)
    sessions.append(session); atomic_json(history_path, sessions)
    started = time.perf_counter(); finished = resumed; last_log = 0.
    try:
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                                 initializer=worker_init, initargs=(str(r.REFERENCE.resolve()),)) as pool:
            iterator = iter(pending); futures = {}
            def submit():
                task = next(iterator, None)
                if task:
                    name, batch = task
                    futures[pool.submit(build_shard, str(output), str(Path(extracted).resolve()), name, batch, fingerprint)] = name
            for _ in range(workers):
                submit()
            while futures:
                done, _ = wait(futures, timeout=20, return_when=FIRST_COMPLETED)
                for future in done:
                    name = futures.pop(future); result = future.result()
                    finished += result['samples']; session['completed_samples'] += result['samples']
                    session['worker_cpu_seconds'] += result['cpu_seconds']
                    submit()
                elapsed = time.perf_counter() - started
                if done or elapsed - last_log >= 30:
                    speed = session['completed_samples'] / max(elapsed, 1e-8)
                    LOG.info('Cache %d/%d (%.2f%%) | elapsed=%.1fs | %.2f img/s | ETA=%.1fs',
                             finished, len(rows), 100 * finished / len(rows), elapsed, speed,
                             (len(rows) - finished) / speed if speed else float('inf'))
                    last_log = elapsed; session['wall_seconds'] = elapsed
                    atomic_json(history_path, sessions)
        session.update(status='COMPLETE', wall_seconds=time.perf_counter() - started, completed_at_utc=r.utc_now())
    except BaseException as error:
        session.update(status='FAILED', wall_seconds=time.perf_counter() - started, failures=1, error=str(error))
        raise
    finally:
        atomic_json(history_path, sessions)
    # Build the catalog only from committed, verified sidecars, in manifest order.
    shards = []; items = []
    for name, batch in plan:
        sidecar = verify_shard(output, name, batch, fingerprint)
        shards.append(dict(name=name, samples=len(batch), sha256=sidecar['file_sha256'], bytes=sidecar['bytes']))
        items.extend(sidecar['items'])
    seconds = sum(s['wall_seconds'] for s in sessions)
    size = sum(s['bytes'] for s in shards)
    if size != context['expected_storage']['npy_bytes']:
        raise ValueError('Actual/expected NPY storage mismatch')
    meta = dict(status='COMPLETE', format_version=VERSION, context_fingerprint=fingerprint,
                created_at_utc=read(context_path)['created_at_utc'], completed_at_utc=r.utc_now(),
                expected_samples=len(rows), samples=len(items), counts=dict(Counter(x['split'] for x in items)),
                shards=shards, items=items, generation_active_wall_seconds=seconds,
                generation_images_per_second=len(rows) / seconds if seconds else None,
                npy_bytes=size, expected_npy_bytes=context['expected_storage']['npy_bytes'],
                failures=sum(s['failures'] for s in sessions), retries=sum(s['retries'] for s in sessions),
                interrupted_sessions=sum(s['status'] == 'INTERRUPTED' for s in sessions),
                generation_timing_is_lower_bound=any(s.get('timing_is_lower_bound', False) for s in sessions),
                cpu_percent_one_core=100 * sum(s['worker_cpu_seconds'] for s in sessions) / max(seconds, 1e-8),
                average_payload_write_bytes_per_second=size / max(seconds, 1e-8),
                disk_throughput_scope='NPY bytes / generation wall time; not a block-device utilization measurement',
                source_images_modified=False, VAL2_accessed=False)
    atomic_json(output / 'cache.json', meta)
    return meta


class FullCacheReader:
    """Bounded read-only mmap LRU; exact manifest binding and per-access SHA check."""
    def __init__(self, root, rows=None, verify_all=False):
        self.root = Path(root); self.meta = read(self.root / 'cache.json')
        self.context = read(self.root / 'context.json')
        rows = frozen_rows() if rows is None else rows
        context = self.context['context']
        if self.context['fingerprint'] != digest(context) or self.meta['context_fingerprint'] != self.context['fingerprint']:
            raise ValueError('Cache context fingerprint mismatch')
        if self.meta['status'] != 'COMPLETE' or self.meta['format_version'] != VERSION:
            raise ValueError('Incomplete/unsupported cache')
        if context['row_mapping_sha256'] != digest(rows):
            raise ValueError('Manifest mapping changed')
        if self.meta['samples'] != len(rows) or self.meta['expected_samples'] != len(rows):
            raise ValueError('Manifest/cache count mismatch')
        self.items = self.meta['items']; self.arrays = OrderedDict(); self.lock = threading.RLock()
        plan = plan_shards(rows, context['shard_size']); offset = 0
        if [s['name'] for s in self.meta['shards']] != [n for n, _ in plan]:
            raise ValueError('Shard inventory mismatch')
        if {p.name for p in self.root.glob('*.npy')} != {name for name, _ in plan}:
            raise ValueError('Orphan or missing shard')
        if any(self.root.glob('*.partial')) or any(self.root.glob('*.json.tmp')):
            raise ValueError('Incomplete cache transaction')
        for (name, batch), descriptor in zip(plan, self.meta['shards']):
            portion = self.items[offset:offset + len(batch)]
            validate_items(portion, batch, name)
            sidecar = read(self.root / (name + '.json'))
            if sidecar['status'] != 'COMPLETE' or sidecar['context_fingerprint'] != self.context['fingerprint'] or sidecar['items'] != portion or sidecar['file_sha256'] != descriptor['sha256']:
                raise ValueError('Catalog/sidecar mismatch')
            array = self._array(name)
            if array.shape != (len(batch), *c.SHAPE) or array.dtype != np.uint8:
                raise ValueError('Shard shape/dtype mismatch')
            if (self.root / name).stat().st_size != array.offset + array.nbytes or descriptor['bytes'] != array.offset + array.nbytes:
                raise ValueError('Shard byte-size mismatch')
            if verify_all:
                verify_shard(self.root, name, batch, self.context['fingerprint'])
            offset += len(batch)
        if offset != len(self.items) or len({x['sample_id'] for x in self.items}) != len(rows):
            raise ValueError('Duplicate or orphan cache entry')
        if self.meta['counts'] != dict(Counter(x['split'] for x in rows)):
            raise ValueError('Split counts mismatch')
        if self.meta['npy_bytes'] != sum(x['bytes'] for x in self.meta['shards']) or self.meta['expected_npy_bytes'] != self.meta['npy_bytes']:
            raise ValueError('Total storage metadata mismatch')
        self.by_id = {x['sample_id']: i for i, x in enumerate(self.items)}

    def _array(self, name):
        path = self.root / name
        if Path(name).name != name or path.is_symlink():
            raise ValueError('Unsafe shard path')
        with self.lock:
            if name not in self.arrays:
                self.arrays[name] = np.load(path, mmap_mode='r', allow_pickle=False)
                if len(self.arrays) > 8:
                    self.arrays.popitem(last=False)
            self.arrays.move_to_end(name)
            return self.arrays[name]

    def get(self, index):
        item = self.items[int(index)]
        pixels = np.array(self._array(item['shard'])[item['index']], copy=True)
        if hashlib.sha256(pixels.tobytes()).hexdigest() != item['tensor_sha256']:
            raise ValueError('Tensor checksum mismatch')
        return pixels

    def indices(self, split, epoch=None):
        if split not in COUNTS:
            raise ValueError('Unknown split')
        result = [i for i, x in enumerate(self.items) if x['split'] == split]
        return sorted(result, key=lambda i: r.rank_id(self.items[i]['sample_id'], f'VAL1-epoch-v1|42|{epoch}|')) if epoch is not None else result


def audit_indices(rows, items, count=1000):
    strata = {}
    ranked = sorted(range(len(rows)), key=lambda i: r.rank_id(rows[i]['sample_id'], 'VAL1-stage2g-strata|42|'))
    for i in ranked:
        row = rows[i]
        strata.setdefault((row['split'], row['class_name'], row['width'], row['height']), i)
    representative = sorted(set(strata.values()) | {i for i in range(len(items)) if items[i]['index'] == 0})
    random = sorted(range(len(rows)), key=lambda i: r.rank_id(rows[i]['sample_id'], 'VAL1-stage2g-random|42|'))[:count]
    return representative, random


def verify(root, extracted):
    started = time.perf_counter(); rows = frozen_rows()
    reader = FullCacheReader(root, rows, verify_all=True)
    context = reader.context['context']
    if digest(environment_contract()) != digest(context['contract']):
        raise ValueError('Verification preprocessing environment changed')
    for path, expected in context['protected_sha256'].items():
        r.sha256_file(Path(path), expected)
    LOG.info('All %d cached tensor checksums and shard mappings verified', len(rows))
    # Only paths enumerated by authenticated VAL1 rows are opened.
    for i, (row, item) in enumerate(zip(rows, reader.items)):
        path = r.image_path(row, Path(extracted)); stat = path.stat()
        if stat.st_size != item['source_size_bytes'] or stat.st_mtime_ns != item['source_mtime_ns']:
            raise ValueError('Source size/mtime changed since caching')
        r.sha256_file(path, row['file_sha256'])
        if (i + 1) % 1000 == 0:
            LOG.info('Source immutability %d/%d', i + 1, len(rows))
    representative, random = audit_indices(rows, reader.items)
    normalizers = r.normalizers(r.REFERENCE); audited = []
    for n, i in enumerate(sorted(set(representative) | set(random))):
        row = rows[i]; pixels = reader.get(i)
        online = c.normalized_uint8(row, r.image_path(row, Path(extracted)), normalizers)
        original_float = r.preprocess_legacy(row, r.image_path(row, Path(extracted)), normalizers)
        if not np.array_equal(pixels, online) or not np.array_equal(c.to_float(pixels), original_float):
            raise ValueError(f'Online/cache exact parity failed: {row["sample_id"]}')
        audited.append(row['sample_id'])
        if (n + 1) % 50 == 0:
            LOG.info('Exact online parity %d/%d', n + 1, len(set(representative) | set(random)))
    result = dict(status='PASS', samples=len(rows), counts=reader.meta['counts'],
                  full_integrity='PASS', source_immutability='PASS', source_checksums_verified=len(rows),
                  representative=dict(status='PASS', samples=len(representative), indices=representative),
                  random_audit=dict(status='PASS', samples=len(random), indices=random, namespace='VAL1-stage2g-random|42|'),
                  audited_sample_ids=audited, exact_uint8=True, exact_float32=True,
                  max_absolute_difference=0, differing_elements=0,
                  seconds=time.perf_counter() - started, completed_at_utc=r.utc_now(),
                  cache_catalog_sha256=r.sha256_file(Path(root) / 'cache.json'),
                  VAL2_accessed=False, source_images_modified=False)
    atomic_json(Path(root) / 'verification.json', result)
    return result


def pilot(extracted, output, workers):
    """Prove multiprocess preprocessing matches the existing Stage 2E cache."""
    rows, _, audit = preflight(extracted, output)
    by_id = {x['sample_id']: x for x in rows}
    baseline = c.CacheReader(Path('runs/stage2e_profile/cache'))
    selected = [by_id[x['sample_id']] for x in baseline.items[:128]]
    root = Path(output).parent / 'pilot'
    root.mkdir(exist_ok=False)
    started = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'),
                             initializer=worker_init, initargs=(str(r.REFERENCE.resolve()),)) as pool:
        jobs = [pool.submit(build_shard, str(root), str(Path(extracted).resolve()), f'pilot_{i:02d}.npy',
                            selected[i * 32:(i + 1) * 32], 'pilot') for i in range(4)]
        for job in jobs:
            job.result()
    for i in range(4):
        data = np.load(root / f'pilot_{i:02d}.npy', mmap_mode='r')
        for j, pixels in enumerate(data):
            if not np.array_equal(pixels, baseline.get(i * 32 + j)):
                raise ValueError('Parallel preprocessing differs from Stage 2E; full generation forbidden')
    result = dict(status='PASS', exact=True, samples=128, workers=workers,
                  seconds=time.perf_counter() - started, preflight=audit)
    atomic_json(root / 'result.json', result)
    LOG.info('Parallel pilot PASS: 128/128 exact Stage 2E tensor matches')
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--extracted', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=4, choices=range(1, 9))
    parser.add_argument('--verify-only', action='store_true')
    parser.add_argument('--preflight-only', action='store_true')
    parser.add_argument('--pilot', action='store_true', help='Run bounded Stage 2E parity pilot only')
    args = parser.parse_args(argv)
    if args.preflight_only:
        _, _, audit = preflight(args.extracted, args.output)
        print(json.dumps(audit, indent=2)); return
    # Logging parent is distinct from the transactional cache directory.
    args.output.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(args.output.parent / 'cache_generation.log')])
    try:
        if args.pilot:
            pilot(args.extracted, args.output, args.workers)
            return
        if not args.verify_only:
            generate(args.extracted, args.output, args.workers)
        verify(args.output, args.extracted)
    except BaseException:
        LOG.exception('Stage 2G cache failed; no PASS claim')
        raise


if __name__ == '__main__':
    main()
