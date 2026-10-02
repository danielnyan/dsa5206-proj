"""Resume VAL1 cache preparation, then run bounded readiness checks only."""
from __future__ import annotations
import argparse
import json
import logging
from pathlib import Path
import subprocess
import sys


def read(path):
    return json.loads(path.read_text())


def launch(root, name, command):
    logging.info('Starting %s', name)
    with (root / (name + '.console.log')).open('a') as stream:
        process = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT)
        while process.poll() is None:
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                logging.info('%s running; see %s.console.log', name, name)
    if process.returncode:
        raise RuntimeError(f'{name} failed ({process.returncode}); no production GO')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, default=Path('runs/stage2g_full_cache'))
    parser.add_argument('--extracted', type=Path, required=True)
    args = parser.parse_args(); root = args.run
    logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s',
                        handlers=[logging.StreamHandler(), logging.FileHandler(root / 'readiness.log')])
    cache = root / 'cache'
    # Kernel-owned lock is released automatically if this runner is interrupted.
    # It does not rely on stale PID files and protects against a second supervisor.
    import fcntl
    with (root / 'readiness.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if not (cache / 'verification.json').exists():
            launch(root, 'cache_resume', [sys.executable, '-B', '-m', 'modern_pca.full_cache',
                   '--output', str(cache), '--extracted', str(args.extracted), '--workers', '4'])
        if read(cache / 'verification.json')['status'] != 'PASS':
            raise ValueError('Cache verification failed')
        for name, command in [
            ('loader', [sys.executable, '-B', 'tools/profile_full_cache.py', '--cache', str(cache), '--output', str(root / 'loader')]),
            ('dry_run', [sys.executable, '-B', '-m', 'modern_pca.cached_training', '--cache', str(cache), '--output', str(root / 'dry_run')])]:
            target = root / name
            if target.exists():
                raise ValueError(f'Preserve/review existing {target} before rerunning; no silent overwrite')
            launch(root, name, command)
            if read(target / 'result.json')['status'] != 'PASS':
                raise ValueError(f'{name} did not pass')
        subprocess.run([sys.executable, '-B', 'tools/report_stage2g.py', '--run', str(root)], check=True)
        logging.info('Stage 2G evidence/report complete. No production training started.')


if __name__ == '__main__':
    main()
