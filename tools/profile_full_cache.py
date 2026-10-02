"""Read the complete verified VAL1 cache through the trainer's actual input path."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys
import time
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from modern_pca import cached_training as t
from modern_pca import full_cache as f
from modern_pca import preprocessing_cache as c
from modern_pca import profile_reimplementation as e
from modern_pca import reimplementation as r


def benchmark(root, output):
    output = Path(output); r.logging_setup(output)
    e.configure('cpu')
    started = time.perf_counter(); reader = t.checked_reader(root)
    startup = time.perf_counter() - started
    # Initialization warmup only; never drop a measured sample from either split.
    for images, _, _ in t.batches(reader, 'train', 1, limit=32):
        images.numpy()
    results = {}
    for split in f.COUNTS:
        expected = reader.indices(split, 1 if split == 'train' else None)
        started = time.perf_counter(); observed = []; batches = 0
        for images, labels, indices in t.batches(reader, split, 1):
            pixels = images.numpy(); codes = labels.numpy(); positions = indices.numpy().tolist()
            if not np.isfinite(pixels).all() or pixels.dtype != np.float32:
                raise ValueError('Invalid loader values')
            if codes.tolist() != [reader.items[i]['ground_truth_code'] for i in positions]:
                raise ValueError('Loader labels differ from manifest')
            observed.extend(positions); batches += 1
            if len(observed) % 1000 == 0:
                seconds = time.perf_counter() - started
                r.LOG.info('%s loader %d/%d | %.2f images/s', split, len(observed), len(expected), len(observed) / seconds)
        seconds = time.perf_counter() - started
        if observed != expected:
            raise ValueError('Loader omitted, duplicated or reordered samples')
        results[split] = dict(samples=len(observed), batches=batches, seconds=seconds,
                              images_per_second=len(observed) / seconds, exact_order=True,
                              order_sha256=f.digest([reader.items[i]['sample_id'] for i in observed]))
    seconds = sum(x['seconds'] for x in results.values())
    result = dict(status='PASS', splits=results, samples=sum(x['samples'] for x in results.values()),
                  seconds=seconds, images_per_second=sum(x['samples'] for x in results.values()) / seconds,
                  reader_startup_seconds=startup, physical_batch=4, parallel=1, prefetch=2,
                  augmentation='stateless train flips; no validation flips', per_access_checksum=True,
                  cache_temperature='uncontrolled OS cache after integrity audit; no cold-cache claim',
                  timing_scope='full input iteration, scaling/flips, host materialization, finite/order/label checks; no model',
                  VAL2_accessed=False, source_images_accessed=False)
    f.atomic_json(output / 'result.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); benchmark(args.cache, args.output)


if __name__ == '__main__':
    main()
