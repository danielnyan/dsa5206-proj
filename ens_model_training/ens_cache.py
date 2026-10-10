"""ENS model 2 input cache: VAL1-only uint8 500x500 tensors built in Kaggle-sized parts.

Paper Fig. 2d: the second ENS convnet works at approximately x35 with a 500 px
patch. The released patches are 600x600 px (150 um at x40; Extended Data Fig. 1),
and model 1 resizes the whole patch to 350 px (x23; Extended Data Fig. 3). Model 2
therefore resizes the whole patch to 500 px (40*500/600 ~ x33) with the unchanged
model-1 RGB decode, Pillow LANCZOS, brightness and Macenko steps.

Source JPEGs are read straight from the MD5-verified official VAL1 ZIPs, each
image SHA-256 checked against the frozen manifest. VAL2 archives are never
downloaded or opened here. No TensorFlow import.
"""
from __future__ import annotations

import argparse
from collections import Counter, OrderedDict
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.metadata
import inspect
import io
import logging
import multiprocessing
import os
from pathlib import Path
import shutil
import sys
import threading
import time
import urllib.request
import zipfile

import numpy as np
from PIL import Image

from modern_pca import full_cache as f
from modern_pca import preprocessing_cache as c
from modern_pca import reimplementation as r

SIZE = 500
SHAPE = (SIZE, SIZE, 3)
VERSION = 'ENS-model2-VAL1-uint8-500-v1'
SHARD_SIZE = 256
PART_SHARDS = 72  # <= 13.83 GB per part, under Kaggle's ~20 GB /kaggle/working output cap
ZENODO = 'https://zenodo.org/records/3825933/files/{}?download=1'
# VAL1 only (name, MD5, bytes from Stage 2A). VAL2 stays locked: its archives are deliberately absent.
ARCHIVES = {'benign': ('val_dataset_1_norm.zip', '1dda32f59996640d701b591f69353115', 7422740863),
            'tumour': ('val_dataset_1_tu.zip', '35494acbfea3adbc689cc15f73a9116d', 4934331128)}
SOFTWARE = ('numpy', 'Pillow', 'staintools', 'spams-bin', 'opencv-python', 'opencv-python-headless',
            'scikit-learn', 'scipy')
LOG = logging.getLogger('ens_cache')


def md5_file(path):
    digest = hashlib.md5()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1 << 24), b''):
            digest.update(block)
    return digest.hexdigest()


def fetch(url, partial, attempts=5):
    """Download with HTTP Range resume; a server ignoring Range restarts the file."""
    for attempt in range(1, attempts + 1):
        offset = partial.stat().st_size if partial.exists() else 0
        request = urllib.request.Request(url, headers={'Range': f'bytes={offset}-'} if offset else {})
        try:
            with urllib.request.urlopen(request, timeout=120) as response:
                resumed = offset and getattr(response, 'status', None) == 206
                with partial.open('ab' if resumed else 'wb') as stream:
                    written, logged = (offset if resumed else 0), time.monotonic()
                    for block in iter(lambda: response.read(1 << 24), b''):
                        stream.write(block); written += len(block)
                        if time.monotonic() - logged >= 60:
                            LOG.info('%s: %.2f GB', partial.name, written / 1e9); logged = time.monotonic()
            return
        except OSError as error:
            LOG.warning('Download attempt %d/%d failed: %s', attempt, attempts, error)
            if attempt == attempts:
                raise


def download(destination, search=()):
    """Return {label: zip path}; reuse an MD5-matching copy under `search`, else fetch from Zenodo."""
    destination = Path(destination); paths = {}
    for label, (name, expected, _) in ARCHIVES.items():
        candidates = [destination / name] + [p for root in search for p in sorted(Path(root).rglob(name))]
        for candidate in candidates:
            if candidate.is_file() and md5_file(candidate) == expected:
                LOG.info('MD5 verified: %s', candidate); paths[label] = candidate; break
        else:
            destination.mkdir(parents=True, exist_ok=True)
            partial = destination / (name + '.partial')
            LOG.info('Downloading %s from Zenodo 3825933', name)
            fetch(ZENODO.format(name), partial)
            actual = md5_file(partial)
            if actual != expected:
                raise ValueError(f'MD5 mismatch for {name}: {actual} != {expected}; archive not used')
            partial.replace(destination / name)
            LOG.info('MD5 verified: %s', destination / name); paths[label] = destination / name
    return paths


class Payload:
    """Path stand-in for evaluate_paper.preprocess_legacy, which only calls read_bytes()."""
    def __init__(self, payload):
        self.payload = payload

    def read_bytes(self):
        return self.payload


class ZipSource:
    """Authenticated VAL1 JPEG bytes from the official ZIPs; same identity rules as r.image_path."""
    def __init__(self, paths):
        self.archives = {label: zipfile.ZipFile(Path(paths[label])) for label in ARCHIVES}

    @staticmethod
    def member(row):
        label = row['class_name']
        if row['cohort'] != 'VAL1':
            raise ValueError('VAL2 is forbidden in training/development/benchmarking')
        if label not in r.CLASSES or row['ground_truth_code'] != r.CLASSES[label]:
            raise ValueError('Invalid class mapping')
        relative = r.normalized_path(row['relative_path'])
        if not relative.startswith(label + '/') or row['sample_id'] != 'VAL1:' + relative:
            raise ValueError('Invalid image identity/path')
        return relative.split('/', 1)[1]  # Stage 2A: archive member == path below the class prefix

    def read(self, row):
        payload = self.archives[row['class_name']].read(self.member(row))
        if hashlib.sha256(payload).hexdigest() != row['file_sha256']:
            raise ValueError(f'Source JPEG checksum mismatch: {row["sample_id"]}')
        return payload

    def check(self, rows):
        names = {label: set(archive.namelist()) for label, archive in self.archives.items()}
        missing = [row['sample_id'] for row in rows if self.member(row) not in names[row['class_name']]]
        if missing:
            raise ValueError(f'{len(missing)} manifest images absent from the ZIPs, e.g. {missing[:3]}')


def model2_pixels(row, payload, normalizers, size=SIZE):
    """evaluate_paper.preprocess_legacy with only the resize target changed; uint8 before /255."""
    if hashlib.sha256(payload).hexdigest() != row['file_sha256']:
        raise ValueError('Source JPEG checksum changed')
    with Image.open(io.BytesIO(payload)) as image:
        rgb = image.convert('RGB')  # No EXIF transpose or colour-profile conversion.
    pixels = np.array(rgb.resize((size, size), Image.Resampling.LANCZOS))
    standardizer, normalizer = normalizers
    pixels = normalizer.transform(standardizer.transform(pixels))
    if pixels.dtype != np.uint8 or pixels.shape != (size, size, 3):
        raise ValueError('Macenko did not return native uint8 RGB; never quantize implicitly')
    return pixels


def to_float(pixels):
    if pixels.dtype != np.uint8 or pixels.ndim != 3:
        raise ValueError('Expected one uint8 HxWx3 cached tensor')
    return np.asarray(pixels, dtype=np.float32) / np.float32(255.0)


def tensor_sha(pixels):
    return hashlib.sha256(np.ascontiguousarray(pixels).tobytes()).hexdigest()


def plan_parts(rows):
    shards = f.plan_shards(rows, SHARD_SIZE)
    return [shards[i:i + PART_SHARDS] for i in range(0, len(shards), PART_SHARDS)]


def shard_bytes(count):
    stream = io.BytesIO()
    np.lib.format.write_array_header_1_0(stream, dict(descr='|u1', fortran_order=False, shape=(count, *SHAPE)))
    return len(stream.getvalue()) + count * int(np.prod(SHAPE))


def canary_rows(rows, count=4):
    """Fixed per-class images whose tensors every part must reproduce bit-exactly."""
    return [row for label in r.CLASSES for row in sorted(
        (x for x in rows if x['class_name'] == label),
        key=lambda x: r.rank_id(x['sample_id'], 'VAL1-ens-canary|42|'))[:count]]


def software():
    versions = {'python': sys.version.split()[0]}
    for name in SOFTWARE:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def context(rows, canaries):
    """Everything that must be identical across parts built in different sessions."""
    return dict(format_version=VERSION, shape=list(SHAPE), shard_size=SHARD_SIZE, part_shards=PART_SHARDS,
                row_mapping_sha256=f.digest(rows), split_sha256=c.SPLIT_HASHES,
                stain_reference_sha256=r.REFERENCE_SHA, software=software(),
                implementation_sha256=hashlib.sha256(inspect.getsource(model2_pixels).replace('\r\n', '\n').encode()).hexdigest(),
                preprocessing=dict(r.preprocessing_settings(), resize=f'Pillow LANCZOS {SIZE}x{SIZE} of the whole native patch',
                                   dtype='uint8', cache_position='after Macenko; before float32/255 and flips'),
                canary_tensor_sha256=canaries)


_WORKER = None


def worker_init(paths, reference):
    global _WORKER
    _WORKER = (ZipSource(paths), r.normalizers(Path(reference)))


def build_shard(root, name, rows):
    source, normalizers = _WORKER
    path = Path(root) / name; partial = Path(root) / (name + '.partial')
    array = np.lib.format.open_memmap(partial, mode='w+', dtype='uint8', shape=(len(rows), *SHAPE))
    items = []
    for index, row in enumerate(rows):
        pixels = model2_pixels(row, source.read(row), normalizers)
        array[index] = pixels
        items.append(dict(sample_id=row['sample_id'], split=row['split'], class_name=row['class_name'],
                          ground_truth_code=row['ground_truth_code'], source_jpeg_sha256=row['file_sha256'],
                          tensor_sha256=tensor_sha(pixels), shard=name, index=index))
    array.flush(); del array
    with partial.open('r+b') as stream:  # Writable handle: fsync on a read-only one fails on Windows.
        os.fsync(stream.fileno())
    partial.replace(path)
    return dict(name=name, samples=len(rows), sha256=r.sha256_file(path), bytes=path.stat().st_size, items=items)


def check_items(items, rows, name):
    keys = ('sample_id', 'split', 'class_name', 'ground_truth_code')
    if len(items) != len(rows) or any(
            [item[k] for k in keys] != [row[k] for k in keys] or item['source_jpeg_sha256'] != row['file_sha256']
            or item['shard'] != name or item['index'] != index for index, (item, row) in enumerate(zip(items, rows))):
        raise ValueError(f'Shard/manifest mapping mismatch: {name}')
    if any(row['cohort'] != 'VAL1' for row in rows):
        raise ValueError('VAL2 is forbidden in the model-2 cache')


def check_free(requirements):
    """requirements: {path: bytes}; paths on one filesystem must fit together."""
    by_device = {}
    for path, needed in requirements.items():
        path = Path(path)
        while not path.exists():
            path = path.parent
        device = os.stat(path).st_dev
        free, total = shutil.disk_usage(path).free, by_device.get(device, (path, 0))[1] + needed
        by_device[device] = (path, total)
        if free < total:
            raise ValueError(f'Insufficient disk at {path}: free {free / 1e9:.1f} GB < needed {total / 1e9:.1f} GB')


def build_part(part, archives, output, workers=4, search=()):
    rows = f.frozen_rows(); plan = plan_parts(rows)
    if not 0 <= part < len(plan):
        raise ValueError(f'Part must be 0..{len(plan) - 1}')
    output = Path(output)
    if output.exists() and any(output.iterdir()):
        raise ValueError(f'Refusing to write into nonempty {output}')
    size = sum(shard_bytes(len(batch)) for _, batch in plan[part])
    zips = sum(size for name, _, size in ARCHIVES.values()
               if not any(any(Path(root).rglob(name)) for root in [archives, *search] if Path(root).exists()))
    check_free({output: size + 200_000_000, archives: zips})
    paths = download(archives, search)
    source = ZipSource(paths); source.check(rows)
    normalizers = r.normalizers(r.REFERENCE)
    canaries = {row['sample_id']: tensor_sha(model2_pixels(row, source.read(row), normalizers)) for row in canary_rows(rows)}
    value = context(rows, canaries); fingerprint = f.digest(value)
    LOG.info('Context fingerprint %s; part %d/%d: %d shards, %.2f GB', fingerprint, part, len(plan) - 1, len(plan[part]), size / 1e9)
    output.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter(); shards = []
    with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context('spawn'), initializer=worker_init,
                             initargs=({k: str(v) for k, v in paths.items()}, str(r.REFERENCE.resolve()))) as pool:
        futures = [pool.submit(build_shard, str(output), name, batch) for name, batch in plan[part]]
        for future, (name, batch) in zip(futures, plan[part]):
            shards.append(future.result())
            done = sum(s['samples'] for s in shards); elapsed = time.perf_counter() - started
            LOG.info('Part %d: %d/%d images | %.2f img/s | %.0fs', part, done, sum(len(b) for _, b in plan[part]), done / elapsed, elapsed)
    for descriptor, (name, batch) in zip(shards, plan[part]):  # Reread every committed tensor.
        check_items(descriptor['items'], batch, name)
        array = np.load(output / name, mmap_mode='r', allow_pickle=False)
        if array.shape != (len(batch), *SHAPE) or descriptor['bytes'] != shard_bytes(len(batch)) or any(
                tensor_sha(pixels) != item['tensor_sha256'] for pixels, item in zip(array, descriptor['items'])):
            raise ValueError(f'Shard verification failed: {name}')
        del array
    seconds = time.perf_counter() - started
    meta = dict(status='COMPLETE', part=part, parts=len(plan), fingerprint=fingerprint, context=value, shards=shards,
                samples=sum(s['samples'] for s in shards), bytes=sum(s['bytes'] for s in shards), seconds=seconds,
                images_per_second=sum(s['samples'] for s in shards) / seconds, workers=workers,
                created_at_utc=r.utc_now(), VAL2_accessed=False, source_images_modified=False)
    f.atomic_json(output / 'ens_part.json', meta)  # Commit marker, written last.
    LOG.info('Part %d COMPLETE: %d images in %.2f h', part, meta['samples'], seconds / 3600)
    return meta


class Model2Cache:
    """Read-only view over all committed parts; every tensor SHA-256 checked on access."""
    def __init__(self, roots, rows=None):
        rows = f.frozen_rows() if rows is None else rows
        plan = plan_parts(rows); parts = {}
        for path in sorted({p.resolve() for root in roots for p in Path(root).rglob('ens_part.json')}):
            meta = f.read(path)
            if meta.get('status') != 'COMPLETE' or meta['part'] in parts:
                raise ValueError(f'Incomplete or duplicate cache part: {path}')
            parts[meta['part']] = (path.parent, meta)
        if sorted(parts) != list(range(len(plan))):
            raise ValueError(f'Cache parts found {sorted(parts)}; all of 0..{len(plan) - 1} must be attached')
        fingerprints = {meta['fingerprint'] for _, meta in parts.values()}
        context = parts[0][1]['context']
        if len(fingerprints) != 1 or f.digest(context) not in fingerprints:
            raise ValueError('Cache parts were built under different preprocessing contexts/software')
        if context['format_version'] != VERSION or context['row_mapping_sha256'] != f.digest(rows):
            raise ValueError('Cache format or frozen VAL1 split mapping mismatch')
        self.context, self.items, self.paths, self.arrays = context, [], {}, OrderedDict()
        self.lock = threading.Lock()  # The trainer reads through a prefetch thread.
        for index, shards in enumerate(plan):
            root, meta = parts[index]
            if [s['name'] for s in meta['shards']] != [name for name, _ in shards]:
                raise ValueError(f'Shard inventory mismatch in part {index}')
            for descriptor, (name, batch) in zip(meta['shards'], shards):
                check_items(descriptor['items'], batch, name)
                path = root / name
                if not path.is_file() or not path.stat().st_size == descriptor['bytes'] == shard_bytes(len(batch)):
                    raise ValueError(f'Missing or truncated shard: {path}')
                self.paths[name] = path; self.items.extend(descriptor['items'])
        if len({x['sample_id'] for x in self.items}) != len(rows) or Counter(x['split'] for x in self.items) != Counter(x['split'] for x in rows):
            raise ValueError('Duplicate, missing or split-crossing cache entry')
        self.fingerprint = f.digest([fingerprints.pop()] + [s['sha256'] for _, m in sorted(parts.items()) for s in m[1]['shards']])

    def _array(self, name):
        with self.lock:
            if name not in self.arrays:
                self.arrays[name] = np.load(self.paths[name], mmap_mode='r', allow_pickle=False)
                if self.arrays[name].dtype != np.uint8 or self.arrays[name].shape[1:] != SHAPE:
                    raise ValueError(f'Invalid shard dtype/shape: {name}')
                if len(self.arrays) > 16:
                    self.arrays.popitem(last=False)
            self.arrays.move_to_end(name)
            return self.arrays[name]

    def get(self, index):
        item = self.items[int(index)]
        array = self._array(item['shard'])  # memmap: validated shape and header offset only
        # One contiguous read per tensor. Slicing the memmap page-faulted 4 KiB at a time, which on
        # NSCC's Lustre scratch starved the GPU (~7 img/s); explicit reads measured ~5000 tensors/s.
        offset = array.offset + int(item['index']) * array.strides[0]
        with open(self.paths[item['shard']], 'rb') as stream:
            stream.seek(offset)
            buffer = stream.read(array.strides[0])
        if len(buffer) != array.strides[0]:
            raise ValueError(f'Short read in shard {item["shard"]} at {offset}')
        pixels = np.frombuffer(buffer, np.uint8).reshape(SHAPE).copy()
        if tensor_sha(pixels) != item['tensor_sha256']:
            raise ValueError(f'Tensor checksum mismatch: {item["sample_id"]}')
        return pixels

    def indices(self, split, epoch=None):
        """Model-1 orders: VAL1-epoch-v1 hash ranks for training, manifest order for validation."""
        if split not in f.COUNTS:
            raise ValueError('Unknown split')
        result = [i for i, x in enumerate(self.items) if x['split'] == split]
        if epoch is None:
            return result
        return sorted(result, key=lambda i: r.rank_id(self.items[i]['sample_id'], f'VAL1-epoch-v1|42|{epoch}|'))


def main(argv=None):
    parser = argparse.ArgumentParser(description='Build one committed model-2 cache part (CPU session)')
    parser.add_argument('--part', type=int, required=True)
    parser.add_argument('--archives', type=Path, required=True, help='Scratch directory for the VAL1 ZIPs')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--workers', type=int, default=os.cpu_count() or 1)
    parser.add_argument('--search', type=Path, nargs='*', default=[], help='Read-only roots that may already hold the ZIPs')
    args = parser.parse_args(argv)
    if not LOG.handlers:
        handler = logging.StreamHandler(sys.stdout)
        handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s'))
        LOG.addHandler(handler); LOG.setLevel(logging.INFO)
    return build_part(args.part, args.archives, args.output, args.workers, args.search)


if __name__ == '__main__':
    main()
