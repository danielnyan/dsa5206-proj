"""Train with a frozen VAL1 stability rule, then evaluate the stopping epoch on VAL2.

The original trainer owns optimizer/RNG recovery. This controller runs one epoch
at a time in a subprocess, archives completed checkpoints, and never consults
VAL2 to stop training. Only trusted checkpoints may be supplied.
"""
import argparse
import csv
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib.metadata
import io
import json
import math
from pathlib import Path
import subprocess
import sys

from . import reimplementation as r


ROOT = Path(__file__).resolve().parents[1]
METRICS = ('train_loss', 'validation_loss', 'train_accuracy', 'validation_accuracy')
DEFAULT_STOPPING = dict(max_epochs=30, min_epochs=10, window=4,
                        loss_tolerance=0.001, accuracy_tolerance=0.001)


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    temporary.replace(path)


def copy_checkpoint(source, target):
    """Atomic immutable snapshot; existing snapshots must be byte-identical."""
    payload = Path(source).read_bytes()
    target = Path(target)
    if target.exists():
        if target.read_bytes() != payload:
            raise ValueError(f'Existing checkpoint differs: {target}')
        return
    temporary = target.with_suffix('.tmp')
    temporary.write_bytes(payload)
    temporary.replace(target)


def validate_policy(policy):
    if not (2 <= policy['window'] <= policy['min_epochs'] <= policy['max_epochs']):
        raise ValueError('Require 2 <= window <= min-epochs <= max-epochs')
    for name in ('loss_tolerance', 'accuracy_tolerance'):
        if not math.isfinite(policy[name]) or policy[name] < 0:
            raise ValueError(f'{name} must be finite and nonnegative')


def stopping_decision(history, policy):
    """Small range over the entire window, not merely small adjacent deltas."""
    validate_policy(policy)
    for expected, row in enumerate(history, 1):
        if row['epoch'] != expected:
            raise ValueError('Epoch history must be contiguous, starting at 1')
        for key in METRICS:
            value = row[key]
            if not math.isfinite(value) or value < 0 or ('accuracy' in key and value > 1):
                raise ValueError(f'Invalid {key} in epoch {expected}')
    epoch = len(history)
    recent = history[-policy['window']:]
    ranges = {key: max(row[key] for row in recent) - min(row[key] for row in recent)
              for key in METRICS} if recent else {}
    stable = (epoch >= policy['min_epochs'] and len(recent) == policy['window']
              and all(ranges[key] <= policy['loss_tolerance' if 'loss' in key else 'accuracy_tolerance']
                      for key in METRICS))
    reason = ('max_epochs' if epoch >= policy['max_epochs'] else
              'train_and_validation_stable' if stable else None)
    return dict(stop=reason is not None, reason=reason, epoch=epoch,
                window_epochs=[row['epoch'] for row in recent], ranges=ranges)


def checkpoint_info(path, batch_size):
    import torch
    import escnn
    payload = Path(path).read_bytes()
    saved = torch.load(io.BytesIO(payload), map_location='cpu', weights_only=False)
    contract, progress = saved['contract'], saved['progress']
    expected = dict(batch_size=batch_size, seed=42, torch=str(torch.__version__),
                    escnn=escnn.__version__, preprocessing='RGB-bilinear350-minus1-plus1',
                    model_sha256=sha256(ROOT / 'modern_pca/group_equivariant_model.py'),
                    split={name: sha256(ROOT / 'manifests/VAL1_internal_v1' / name)
                           for name in ('VAL1_train.csv', 'VAL1_internal_validation.csv')})
    if contract != expected:
        raise ValueError('Checkpoint training contract differs; keep the original code, batch size and environment')
    if progress['epoch'] != len(progress['history']) or progress['next_batch'] < 0:
        raise ValueError('Inconsistent checkpoint progress')
    return dict(sha256=hashlib.sha256(payload).hexdigest(), contract=contract,
                epoch=progress['epoch'], next_batch=progress['next_batch'],
                history=progress['history'])


def current_config(args):
    policy = {key: getattr(args, key) for key in DEFAULT_STOPPING}
    validate_policy(policy)
    return dict(
        schema=1, stopping=policy, selection='stopping_epoch',
        batch_size=args.batch_size, checkpoint_every=args.checkpoint_every, device=args.device,
        data=str(args.data.resolve()),
        code_sha256={name: sha256(ROOT / 'modern_pca' / name) for name in (
            'group_equivariant_model.py', 'train_group_equivariant.py',
            'evaluate_group_equivariant.py', 'run_group_equivariant_experiment.py',
            'reimplementation.py')},
        environment={name: importlib.metadata.version(name) for name in (
            'torch', 'torchvision', 'escnn', 'numpy', 'Pillow', 'scikit-learn')},
        manifest_sha256={name: sha256(ROOT / 'manifests' / name)
                         for name in ('VAL1_manifest.csv', 'VAL2_manifest.csv')},
        prior_val2_exploration=True,
        caveat='VAL2 has previously been evaluated in this project; this is not a newly untouched test cohort.',
    )


def execute(command, log_path):
    """Display and append child output; always stop our own child on interruption."""
    with Path(log_path).open('a', encoding='utf-8') as log:
        with subprocess.Popen(command, cwd=ROOT, stdout=subprocess.PIPE,
                              stderr=subprocess.STDOUT, text=True) as process:
            try:
                for line in process.stdout:
                    print(line, end='', flush=True)
                    log.write(line)
                    log.flush()
                code = process.wait()
                if code:
                    raise subprocess.CalledProcessError(code, command)
            except BaseException:
                if process.poll() is None:
                    process.terminate()
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                raise


def export_history(output, info, policy):
    temporary = output / 'history.csv.tmp'
    with temporary.open('w', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=['epoch', *METRICS])
        writer.writeheader()
        writer.writerows({key: row[key] for key in writer.fieldnames} for row in info['history'])
    temporary.replace(output / 'history.csv')
    write_json(output / 'stopping_status.json', stopping_decision(info['history'], policy))


def run(args):
    args.data, args.output = args.data.resolve(), args.output.resolve()
    if args.resume_from:
        args.resume_from = args.resume_from.resolve()
    r.ensure_output_separation(args.data, [args.output])
    config = current_config(args)
    if args.resume and args.resume_from:
        raise ValueError('Use --resume for an existing experiment OR --resume-from to initialize a new one')
    source = checkpoint_info(args.resume_from, args.batch_size) if args.resume_from else None
    if source:
        stopping_decision(source['history'], config['stopping'])
        if source['epoch'] > args.max_epochs or (source['epoch'] == args.max_epochs and source['next_batch']):
            raise ValueError('Source checkpoint is beyond the maximum epoch target')
    if args.plan:
        if args.resume:
            protocol = json.loads((args.output / 'protocol.json').read_text())
            if protocol['config'] != config:
                raise ValueError('Frozen protocol differs')
            last = args.output / 'training/last.pt'
            source = checkpoint_info(last, args.batch_size) if last.exists() else protocol['initial_checkpoint']
        decision = stopping_decision(source['history'] if source else [], config['stopping'])
        print(json.dumps(dict(config=config, checkpoint=source, stopping_decision=decision,
                              next_training_epoch=None if decision['stop'] else source['epoch'] + 1 if source else 1), indent=2))
        return
    if args.output.exists() and not args.resume:
        raise FileExistsError('Output exists. Use --resume with the same settings, or choose a new output directory')
    if args.resume and not args.output.is_dir():
        raise FileNotFoundError('Experiment directory does not exist')
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / '.experiment.lock').open('a') as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError('Another controller is running in this output directory') from error
        return run_locked(args, config, source)


def run_locked(args, config, source):
    output = args.output
    protocol_path = output / 'protocol.json'
    if args.resume:
        protocol = json.loads(protocol_path.read_text())
        if protocol['config'] != config:
            raise ValueError('Frozen protocol differs (settings, code or environment); use the original configuration')
    else:
        protocol = dict(config=config, created_at_utc=datetime.now(timezone.utc).isoformat(),
                        initial_checkpoint=source,
                        initial_checkpoint_path=str(args.resume_from) if source else None,
                        policy_basis='Thresholds chosen using VAL1 pilot epochs 1-5; no VAL2 metrics used by this controller')
        write_json(protocol_path, protocol)
    policy = config['stopping']
    training = output / 'training'
    training.mkdir(exist_ok=True)
    archives = output / 'epochs'
    archives.mkdir(exist_ok=True)
    last = training / 'last.pt'
    initial = protocol['initial_checkpoint']
    if not last.exists() and initial:
        snapshot = output / 'initial.pt'
        if not snapshot.exists():
            origin = Path(protocol['initial_checkpoint_path'])
            if sha256(origin) != initial['sha256']:
                raise ValueError('Initial checkpoint changed since the protocol was frozen')
            copy_checkpoint(origin, snapshot)
        if sha256(snapshot) != initial['sha256']:
            raise ValueError('Initial checkpoint snapshot changed')
        copy_checkpoint(snapshot, last)
    selection_path = output / 'selection.json'
    if not selection_path.exists():
        while True:
            info = checkpoint_info(last, args.batch_size) if last.exists() else dict(epoch=0, next_batch=0, history=[])
            # Recover an epoch archive even if the controller was interrupted
            # immediately after the trainer committed its completed checkpoint.
            if info['epoch'] > (initial['epoch'] if initial else 0) and info['next_batch'] == 0:
                copy_checkpoint(last, archives / f"epoch_{info['epoch']:03d}.pt")
            export_history(output, info, policy)
            decision = stopping_decision(info['history'], policy)
            if decision['stop'] and info['next_batch'] == 0:
                final = output / 'final.pt'
                copy_checkpoint(last, final)
                selection = dict(**decision, checkpoint='final.pt', checkpoint_sha256=sha256(final),
                                 protocol_sha256=sha256(protocol_path),
                                 selected_at_utc=datetime.now(timezone.utc).isoformat(),
                                 rule='Use the stopping epoch, not the lowest VAL2 loss or best VAL2 accuracy')
                write_json(selection_path, selection)
                break
            if info['epoch'] >= args.max_epochs:
                raise ValueError('A partial checkpoint is beyond the frozen maximum epoch target')
            target = info['epoch'] + 1
            command = [sys.executable, '-u', '-m', 'modern_pca.train_group_equivariant',
                       '--data', str(args.data), '--output', str(training), '--epochs', str(target),
                       '--batch-size', str(args.batch_size), '--device', args.device,
                       '--checkpoint-every', str(args.checkpoint_every)]
            if last.exists():
                command += ['--resume', str(last)]
            print(f'Training through epoch {target}/{args.max_epochs}', flush=True)
            execute(command, output / 'training.log')
            info = checkpoint_info(last, args.batch_size)
            if info['epoch'] != target or info['next_batch'] != 0:
                raise ValueError('Trainer did not finish the requested epoch; VAL2 will not be evaluated')
            copy_checkpoint(last, archives / f'epoch_{target:03d}.pt')
    selection = json.loads(selection_path.read_text())
    final = output / selection['checkpoint']
    if selection['protocol_sha256'] != sha256(protocol_path) or sha256(final) != selection['checkpoint_sha256']:
        raise ValueError('Frozen checkpoint or selection protocol changed')
    final_info = checkpoint_info(final, args.batch_size)
    decision = stopping_decision(final_info['history'], policy)
    if not decision['stop'] or decision['epoch'] != selection['epoch'] or decision['reason'] != selection['reason']:
        raise ValueError('Final checkpoint does not satisfy the frozen stopping rule')

    # Check for a completed evaluator result before starting another process.
    # Failed partial attempts are retained; resuming uses a fresh attempt folder.
    evaluation_root = output / 'val2'
    evaluation_root.mkdir(exist_ok=True)
    completed = list(evaluation_root.glob('attempt_*/metrics.json'))
    if len(completed) > 1:
        raise ValueError('Multiple completed VAL2 evaluations found; inspect the experiment directory')
    if not completed:
        attempt = 1
        while (evaluation_root / f'attempt_{attempt:03d}').exists():
            attempt += 1
        destination = evaluation_root / f'attempt_{attempt:03d}'
        execute([sys.executable, '-u', '-m', 'modern_pca.evaluate_group_equivariant',
                 '--data', str(args.data), '--checkpoint', str(final), '--output', str(destination),
                 '--batch-size', str(args.batch_size), '--device', args.device], output / 'evaluation.log')
        completed = [destination / 'metrics.json']
    evaluation_path = completed[0]
    evaluation = json.loads(evaluation_path.read_text())
    if (evaluation['status'] != 'complete' or evaluation['checkpoint_sha256'] != selection['checkpoint_sha256']
            or evaluation['completed_training_epochs'] != selection['epoch']
            or evaluation['samples'] != 34103 or evaluation['cohort'] != 'VAL2'
            or evaluation['evaluator_sha256'] != config['code_sha256']['evaluate_group_equivariant.py']
            or evaluation['manifest_sha256'] != config['manifest_sha256']['VAL2_manifest.csv']):
        raise ValueError('Completed VAL2 report does not match this experiment')
    summary = dict(status='complete', protocol_sha256=sha256(protocol_path), selection=selection,
                   prior_val2_exploration=True, caveat=config['caveat'],
                   val1_history=final_info['history'], val2_metrics=evaluation['metrics'],
                   evaluation_report=str(evaluation_path.relative_to(output)),
                   evaluation_sha256=sha256(evaluation_path),
                   predictions_sha256=sha256(evaluation_path.parent / 'predictions.csv'))
    summary_path = output / 'summary.json'
    if summary_path.exists() and json.loads(summary_path.read_text()) != summary:
        raise ValueError('Completed experiment artifacts changed')
    write_json(summary_path, summary)
    print(f"Experiment complete: epoch {selection['epoch']}, reason={selection['reason']}", flush=True)
    print(f'Report: {summary_path}', flush=True)
    return summary


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New experiment directory, separate from pilot runs')
    parser.add_argument('--resume-from', type=Path, help='Trusted pilot checkpoint to copy into a new experiment')
    parser.add_argument('--resume', action='store_true', help='Recover an existing experiment with the same protocol')
    parser.add_argument('--plan', action='store_true', help='Inspect configuration/source checkpoint without writing or training')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--checkpoint-every', type=int, default=100)
    parser.add_argument('--device', choices=['cuda', 'cpu'], default='cuda')
    for name, default in DEFAULT_STOPPING.items():
        parser.add_argument('--' + name.replace('_', '-'), type=type(default), default=default)
    args = parser.parse_args(argv)
    if min(args.batch_size, args.checkpoint_every) < 1:
        parser.error('Batch size and checkpoint interval must be positive')
    try:
        validate_policy({name: getattr(args, name) for name in DEFAULT_STOPPING})
    except ValueError as error:
        parser.error(str(error))
    return args


if __name__ == '__main__':
    run(parse_args())
