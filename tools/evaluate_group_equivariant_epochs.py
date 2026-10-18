"""Evaluate archived epochs on VAL2 and export an exploratory comparison.

Uses the existing evaluator unchanged. Existing outputs are never overwritten.
Run from the repository root with the same environment as training.
"""
import argparse
import contextlib
import csv
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import torch

from modern_pca import evaluate_group_equivariant as evaluator


class Tee:
    def __init__(self, *streams):
        self.streams = streams

    def write(self, text):
        for stream in self.streams:
            stream.write(text)

    def flush(self):
        for stream in self.streams:
            stream.flush()


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--baseline', type=Path, required=True,
                        help='Existing epoch-5 metrics.json')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--start-epoch', type=int, default=14)
    parser.add_argument('--end-epoch', type=int, default=20)
    args = parser.parse_args()
    if not 6 <= args.start_epoch <= args.end_epoch:
        parser.error('Require 6 <= start-epoch <= end-epoch (epoch 5 is the baseline)')
    epochs = list(range(args.start_epoch, args.end_epoch + 1))
    torch.set_num_threads(2)
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for this comparison')
    baseline = json.loads(args.baseline.read_text())
    assert baseline['status'] == 'complete'
    assert baseline['completed_training_epochs'] == 5
    assert baseline['checkpoint_sha256'] == sha256(Path(baseline['checkpoint']))
    history = {row['epoch']: row for row in json.loads(
        (args.run / 'summary.json').read_text())['val1_history']}
    checkpoints = [args.run / 'epochs' / f'epoch_{epoch:03d}.pt'
                   for epoch in epochs]
    if any(epoch not in history for epoch in epochs):
        raise ValueError('Requested epochs are missing from the training history')
    checkpoint_hashes = {str(path.resolve()): sha256(path) for path in checkpoints}
    args.output.mkdir(parents=True, exist_ok=False)
    protocol = {
        'started_at_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': f'Exploratory retrospective VAL2 comparison of epochs {args.start_epoch}-{args.end_epoch} with epoch 5',
        'caveat': 'VAL2 has participated in development; these are not untouched final test results. No automatic checkpoint selection or stopping-rule changes.',
        'epochs': epochs, 'batch_size': 4, 'device': 'cuda',
        'torch_num_threads': 2, 'checkpoint_sha256': checkpoint_hashes,
        'baseline_metrics': str(args.baseline.resolve()),
        'baseline_metrics_sha256': sha256(args.baseline),
        'original_summary_sha256': sha256(args.run / 'summary.json'),
        'evaluator_sha256': sha256(Path(evaluator.__file__)),
        'runner_sha256': sha256(Path(__file__)),
        'command': [sys.executable, '-m', 'tools.evaluate_group_equivariant_epochs', *sys.argv[1:]],
    }
    (args.output / 'protocol.json').write_text(json.dumps(protocol, indent=2) + '\n')
    reports = [(args.baseline.resolve(), baseline)]

    def export(status):
        rows = []
        for path, report in reports:
            epoch = report['completed_training_epochs']
            row = {'epoch': epoch, 'source': 'existing_baseline' if epoch == 5 else 'fresh_evaluation',
                   'samples': report['samples'], **history[epoch],
                   **{f'val2_{key}': value for key, value in report['metrics'].items()
                      if key != 'confusion_matrix'},
                   'metrics_path': str(path), 'metrics_sha256': sha256(path),
                   'predictions_sha256': sha256(path.parent / 'predictions.csv')}
            row['val2_loss_delta_vs_epoch5'] = report['metrics']['loss'] - baseline['metrics']['loss']
            row['val2_accuracy_delta_pp_vs_epoch5'] = 100 * (report['metrics']['accuracy'] - baseline['metrics']['accuracy'])
            rows.append(row)
        with (args.output / 'comparison.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        (args.output / 'comparison.json').write_text(json.dumps({
            'status': status, 'protocol': protocol, 'results': rows,
        }, indent=2, allow_nan=False) + '\n')

    export('running')
    for epoch, checkpoint in zip(epochs, checkpoints):
        with (args.output / f'epoch_{epoch:03d}.log').open('x') as stream:
            with contextlib.redirect_stdout(Tee(sys.stdout, stream)):
                print(f'Starting epoch {epoch}', flush=True)
                report = evaluator.run(argparse.Namespace(
                    data=args.data, checkpoint=checkpoint,
                    output=args.output / f'epoch_{epoch:03d}', batch_size=4, device='cuda'))
        assert report['completed_training_epochs'] == epoch
        assert report['checkpoint_sha256'] == checkpoint_hashes[str(checkpoint.resolve())]
        for key in ('samples', 'class_counts', 'manifest_sha256', 'evaluator_sha256',
                    'model_code_sha256', 'preprocessing', 'batch_size', 'decision_rule',
                    'training_contract'):
            assert report[key] == baseline[key], f'Baseline comparison mismatch: {key}'
        reports.append(((args.output / f'epoch_{epoch:03d}' / 'metrics.json').resolve(), report))
        export('running')
    for path, digest in checkpoint_hashes.items():
        assert sha256(Path(path)) == digest
    assert sha256(args.run / 'summary.json') == protocol['original_summary_sha256']
    export('complete')
    print(f'All {len(epochs)} evaluations complete: {args.output / "comparison.csv"}', flush=True)


if __name__ == '__main__':
    main()
