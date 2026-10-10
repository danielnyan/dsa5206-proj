"""ens_train.run_probe on NSCC, optionally before the cache exists.

With --synthetic, run_probe's reader is replaced by fixed random uint8 500x500 tensors (as in the
Kaggle T4 probe of 2026-10-05), so peak memory and img/s exclude cache I/O. Everything else is
run_probe unchanged: real NASNetLarge model 2, real Adam/accumulation updates at the first and
deepest stages, compiled frozen forward pass, 14-epoch forecast. Writes <output>/probe.json.

    python jobs/gpu_probe.py --synthetic --output $RUNS/probe_b4 --batch-size 4
    python -m ens_model_training.ens_train probe --cache-roots $CACHE --output ... --batch-size 4   # real cache
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from ens_model_training import ens_cache as e
from ens_model_training import ens_train as t


class SyntheticCache:
    """Reader stand-in with Model2Cache's interface; nothing from the dataset is read."""
    indices = e.Model2Cache.indices
    fingerprint = 'synthetic'

    def __init__(self, roots=None, train=256, validation=128):
        self.pixels = np.random.default_rng(0).integers(0, 256, (train + validation, *e.SHAPE), dtype=np.uint8)
        self.items = [dict(sample_id=f'VAL1:synthetic/{i:04d}.jpg', ground_truth_code=i % 2,
                           split='train' if i < train else 'internal_validation') for i in range(train + validation)]

    def get(self, index):
        return self.pixels[int(index)].copy()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--batch-size', type=int, choices=[1, 2, 4], required=True)
    parser.add_argument('--synthetic', action='store_true')
    parser.add_argument('--cache-roots', type=Path, nargs='*', default=[])
    args = parser.parse_args(argv)
    if args.synthetic:
        e.Model2Cache = SyntheticCache  # run_probe looks the class up on the module at call time
    elif not args.cache_roots:
        parser.error('--cache-roots required without --synthetic')
    result = t.run_probe(args)
    result['reader'] = 'synthetic uint8 tensors' if args.synthetic else 'Model2Cache'
    t.f.atomic_json(args.output / 'probe.json', result)


if __name__ == '__main__':
    main()
