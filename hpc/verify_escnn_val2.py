"""CPU preflight for the repository's existing Stage-2G D8 VAL2 evaluator.

Use only with trusted, self-generated best.pt checkpoints. Does not train or
select a checkpoint. Tests actual preprocessing for one image of each class.
"""
import argparse
import hashlib
import importlib.metadata
import sys
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--data', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--last', type=Path, required=False)
    args = p.parse_args()

    import cv2
    import matplotlib
    import numpy as np
    import scipy
    import sklearn
    import spams
    import staintools
    import torch
    import escnn
    from modern_pca import reimplementation as r
    from modern_pca.escnn_stage2g import PREPROCESSING

    import modern_pca.evaluate_group_equivariant as evaluator
    root = Path(evaluator.REPOSITORY)
    # Inspect checkpoint contract WITHOUT rebuilding model or allocating GPU.
    # Only trust checkpoint files created by the user's own training job.
    saved = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    progress, contract = saved['progress'], saved['contract']
    expected = dict(
        torch=str(torch.__version__),
        escnn=str(escnn.__version__),
        preprocessing=PREPROCESSING,
        model_sha256=hashlib.sha256(
            (root / 'modern_pca/group_equivariant_model.py').read_bytes()
        ).hexdigest(),
    )
    for name, value in expected.items():
        if contract.get(name) != value:
            raise RuntimeError(f'{name} checkpoint mismatch: {contract.get(name)} != {value}')
    if not contract.get('cache_catalog_sha256'):
        raise RuntimeError('Checkpoint lacks Stage-2G cache provenance')
    if progress['epoch'] < 1 or progress['next_batch'] != 0:
        raise RuntimeError('best.pt was not saved after a completed epoch')
    if args.last is not None and args.last.is_file():
        last = torch.load(args.last, map_location='cpu', weights_only=False)['progress']
        if last['best_epoch'] != progress['epoch']:
            raise RuntimeError('best.pt epoch differs from last.pt best_epoch')

    rows = r.read_manifest(root / 'manifests/VAL2_manifest.csv', cohort='VAL2')
    if len(rows) != 34103:
        raise RuntimeError(f'Unexpected VAL2 cohort count: {len(rows)}')
    # Fail before GPU allocation if VAL2 files or stain processing are unusable.
    normalizers = r.normalizers(root / r.REFERENCE)
    for label in ('benign', 'tumour'):
        row = next(x for x in rows if x['class_name'] == label)
        path = r.image_path(row, args.data, final_evaluation=True)
        if not path.is_file():
            raise FileNotFoundError(f'Missing VAL2 file for {label}: {path}')
        image = r.preprocess_legacy(row, path, normalizers)
        assert image.shape == (350, 350, 3)
        assert image.dtype == np.float32
        assert np.isfinite(image).all() and image.min() >= 0 and image.max() <= 1
        print('VAL2 sample preprocessing PASS:', label, row['sample_id'], flush=True)
    print('Checkpoint best epoch:', progress['epoch'])
    for name in ('numpy', 'scipy', 'scikit-learn', 'torch', 'escnn',
                 'staintools', 'spams-bin', 'opencv-python', 'matplotlib'):
        print(f'{name}={importlib.metadata.version(name)}')
    print('Stage-2G D8 VAL2 CPU preflight PASS', flush=True)


if __name__ == '__main__':
    main()
