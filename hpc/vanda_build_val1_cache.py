"""Generate and verify the repository-format VAL1 Stage-2G cache on Vanda.

Unlike original full_cache.py, the environment contract here records the ACTUAL
Vanda environment rather than demanding an unavailable WSL Stage-2E reference.
It uses the exact original full_cache.generate()/verify() implementations,
including per-source and per-tensor hashes, immutable split, and an independent
online/cache preprocessing parity audit. This is a new Vanda-local verification,
not a claim of cross-machine byte parity with the original WSL cache.

Run from repository root:
python -m hpc.vanda_build_val1_cache --extracted ROOT --output CACHE --workers 4
"""
from __future__ import annotations
import argparse
import hashlib
import importlib
import importlib.metadata as metadata
import json
import platform
import sys
from pathlib import Path


def file_sha256(path):
    with Path(path).open('rb') as stream:
        h = hashlib.sha256()
        for block in iter(lambda: stream.read(1024*1024), b''):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--extracted', required=True, type=Path)
    ap.add_argument('--output', required=True, type=Path)
    ap.add_argument('--workers', type=int, choices=range(1, 9), default=4)
    ap.add_argument('--preflight-only', action='store_true')
    ap.add_argument('--verify-only', action='store_true')
    args = ap.parse_args()
    if args.preflight_only and args.verify_only:
        ap.error('Select one of preflight-only or verify-only')

    # Fail early with useful errors for missing Vanda stain dependencies.
    for name in ('numpy', 'PIL', 'staintools', 'spams', 'cv2', 'tensorflow'):
        try:
            importlib.import_module(name)
        except Exception as exc:
            raise RuntimeError(f'Vanda Python environment cannot import {name}: {exc}') from exc

    from modern_pca import full_cache as cache
    from modern_pca import preprocessing_cache as prep
    from modern_pca import reimplementation as repo

    dependencies = ('numpy', 'pillow', 'staintools', 'spams-bin', 'opencv-python',
                    'tensorflow', 'keras', 'scipy')
    def environment_contract():
        versions = {}
        for name in dependencies:
            try:
                versions[name] = metadata.version(name)
            except metadata.PackageNotFoundError:
                versions[name] = None
        reference = Path(repo.REFERENCE)
        repo.sha256_file(reference, repo.REFERENCE_SHA)
        provenance = prep.cache_provenance()
        # Hardware RAM / CUDA visibility can differ across PBS allocations.
        # Keep software provenance without binding a resumable cache to a node.
        provenance['environment'] = {'software': provenance['environment']['software']}
        return {
            'contract': 'VANDA_VAL1_FULL_CACHE_V1',
            'validation_method': 'full_cache.generate+verify original implementation',
            'machine': {'python': platform.python_version(),
                        'platform': platform.platform()},
            'dependencies': versions,
            'preprocessing_provenance': provenance,
            'source_sha256': {
                'vanda_build_val1_cache.py': file_sha256(__file__),
                'full_cache.py': file_sha256(cache.__file__),
                'preprocessing_cache.py': file_sha256(prep.__file__),
                'reimplementation.py': file_sha256(repo.__file__),
            },
            'reference_sha256': repo.REFERENCE_SHA,
            'separate_from_original_WSL_environment': True,
        }

    # Deliberately substitute only the WSL-specific environment contract.
    # Full preprocessing, source integrity checks, and parity verification are
    # delegated unchanged to the original verified cache implementation.
    cache.environment_contract = environment_contract
    import logging
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
    extracted = args.extracted.resolve()
    output = args.output.resolve()
    print(json.dumps({'mode': 'vanda-local-cache-v1', 'extracted': str(extracted),
                      'output': str(output), 'workers': args.workers,
                      'environment': environment_contract()}, indent=2), flush=True)
    if args.preflight_only:
        _, _, audit = cache.preflight(extracted, output)
        print(json.dumps(audit, indent=2), flush=True)
        return
    if args.verify_only:
        result = cache.verify(output, extracted)
        print('VERIFICATION PASS', result['status'], flush=True)
        return
    cache.generate(extracted, output, workers=args.workers)
    result = cache.verify(output, extracted)
    if result['status'] != 'PASS':
        raise RuntimeError('Full cache verification failed')
    print('VERIFICATION PASS; cache ready for scratch NASNet training', flush=True)


if __name__ == '__main__':
    main()
