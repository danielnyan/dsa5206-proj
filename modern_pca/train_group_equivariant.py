"""Train the genuine D8 ESCNN from scratch on the verified Stage-2G VAL1 cache.

No TensorFlow, no image reprocessing, no gradient accumulation, no VAL2 access.
Resumable during an epoch; best.pt is selected by VAL1 validation loss.
"""

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import time


def atomic_save(state, path):
    """Preserve last complete checkpoint even if PBS interrupts the save."""
    import torch

    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cache', type=Path, required=True,
                        help='Existing verified Stage-2G cache directory (contains cache.json)')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=20,
                        help='Maximum TOTAL completed epochs, including resumes')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--num-workers', type=int, default=4)
    parser.add_argument('--checkpoint-every', type=int, default=1000,
                        help='Save at this many training batches or after 300 seconds')
    parser.add_argument('--patience', type=int, default=4,
                        help='Stop after this many complete epochs without validation loss improvement')
    parser.add_argument('--min-delta', type=float, default=0.,
                        help='Minimum absolute validation-loss improvement; default strict < comparison')
    parser.add_argument('--resume', type=Path, help='Trusted last.pt from this script')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.checkpoint_every, args.patience) < 1:
        parser.error('Epochs, batch size, checkpoint interval, patience must be positive')
    if args.num_workers < 0 or not math.isfinite(args.min_delta) or args.min_delta < 0:
        parser.error('Invalid worker count or min_delta')
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    import escnn
    import numpy as np
    import random
    from torch.utils.data import DataLoader, Subset
    from . import reimplementation as r
    from .escnn_stage2g import load_stage2g_datasets, PREPROCESSING
    from .group_equivariant_model import D8MultiScaleResNetGAP

    repository = Path(__file__).resolve().parents[1]
    r.ensure_output_separation(args.cache, [args.output])
    last = args.output / 'last.pt'
    best = args.output / 'best.pt'
    if last.exists() and args.resume is None:
        raise ValueError('Existing last.pt: pass --resume or choose a new output')
    if args.resume is not None and not last.exists():
        raise FileNotFoundError('Resume requires an existing last.pt')
    if args.resume is not None and args.resume.resolve() != last.resolve():
        raise ValueError('Resume only from this output directory\'s last.pt')

    args.output.mkdir(parents=True, exist_ok=True)
    train_rows, validation_rows, split = r.freeze_split(
        repository / 'manifests/VAL1_manifest.csv', args.output / 'split'
    )
    train_data, validation_data, catalog_sha = load_stage2g_datasets(
        args.cache, train_rows, validation_rows
    )

    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError('CUDA requested but unavailable')

    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)

    model = D8MultiScaleResNetGAP().to(device)
    if (model.gspace.fibergroup.order() != 16
            or [model.f88_type.size, model.f44_type.size,
                model.f22_type.size, model.f11_type.size] != [128, 256, 384, 512]):
        raise RuntimeError('Expected genuine D8 with original field counts')
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.2, patience=2
    )
    criterion = torch.nn.CrossEntropyLoss()
    contract = dict(
        split=split['fingerprints'], batch_size=args.batch_size, seed=42,
        model_sha256=hashlib.sha256(
            (repository / 'modern_pca/group_equivariant_model.py').read_bytes()).hexdigest(),
        trainer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        reader_sha256=hashlib.sha256(
            (repository / 'modern_pca/escnn_stage2g.py').read_bytes()).hexdigest(),
        cache_catalog_sha256=catalog_sha,
        torch=str(torch.__version__), escnn=str(escnn.__version__),
        preprocessing=PREPROCESSING,
        early_stopping=dict(monitor='validation_loss', patience=args.patience,
                            min_delta=args.min_delta, mode='min'),
        optimizer='Adam(lr=0.0001)',
        scheduler='ReduceLROnPlateau(mode=min,factor=0.2,patience=2)',
    )
    state = dict(epoch=0, next_batch=0, loss_sum=0., correct=0, samples=0,
                 history=[], best_val_loss=float('inf'), best_epoch=0,
                 epochs_without_improvement=0)
    if args.resume:
        # Trusted self-generated checkpoint only; weights_only=False is intentional
        # because Python/NumPy RNG state is stored for mid-epoch recovery.
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['contract'] != contract:
            raise ValueError('Checkpoint data/model/optimizer/early stopping contract differs')
        model.train()  # escnn removes derived evaluation filter buffers
        model.load_state_dict(saved['model'], strict=True)
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        state = saved['progress']
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['rng'])
        if device.type == 'cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        if state['best_epoch'] and not best.is_file():
            raise FileNotFoundError('best.pt missing: do not resume a partially lost experiment')
        print(f"Resume: completed epochs={state['epoch']}, next batch={state['next_batch']}", flush=True)

    def save(path=last):
        atomic_save(dict(
            contract=contract, progress=state,
            model=model.state_dict(), optimizer=optimizer.state_dict(),
            scheduler=scheduler.state_dict(),
            rng=torch.get_rng_state(), python_rng=random.getstate(),
            numpy_rng=np.random.get_state(),
            cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None,
        ), path)

    loader_options = dict(num_workers=args.num_workers,
                          pin_memory=(device.type == 'cuda'))
    if args.num_workers:
        loader_options['prefetch_factor'] = 2
    valid_loader = DataLoader(validation_data, batch_size=args.batch_size,
                              shuffle=False, **loader_options)

    print(f'{device}: train={len(train_data)} val={len(validation_data)} '
          f'fields=(8,16,24,32) group_order=16 batch={args.batch_size} '
          f'workers={args.num_workers}', flush=True)
    if not last.exists():
        save()  # preserve initial parameters before consuming any samples

    while (state['epoch'] < args.epochs
           and state['epochs_without_improvement'] < args.patience):
        epoch = state['epoch']
        order = torch.randperm(
            len(train_data), generator=torch.Generator().manual_seed(42 + epoch)
        ).tolist()
        offset = state['next_batch'] * args.batch_size
        # Sample order and batch offsets are stable across PBS resumptions.
        loader = DataLoader(Subset(train_data, order[offset:]),
                            batch_size=args.batch_size, shuffle=False,
                            **loader_options)
        model.train()
        saved_at = time.monotonic()
        started = time.monotonic()
        for images, labels in loader:
            images = images.to(device, non_blocking=True).float().div_(255.0)
            labels = labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; investigate before resuming')
            loss.backward()
            optimizer.step()
            state['next_batch'] += 1
            state['loss_sum'] += loss.item() * len(labels)
            state['correct'] += (logits.argmax(1) == labels).sum().item()
            state['samples'] += len(labels)
            if (state['next_batch'] % args.checkpoint_every == 0
                    or time.monotonic() - saved_at >= 300):
                save()
                saved_at = time.monotonic()
                print(f"Epoch {epoch+1}/{args.epochs}: samples="
                      f"{state['samples']}/{len(train_data)}; checkpoint saved", flush=True)
        # Preserve completed training updates before potentially lengthy validation.
        save()
        model.eval()
        val_loss_sum, val_correct, count = 0., 0, 0
        with torch.inference_mode():
            for images, labels in valid_loader:
                images = images.to(device, non_blocking=True).float().div_(255.0)
                labels = labels.to(device, non_blocking=True)
                logits = model(images)
                val_loss_sum += criterion(logits, labels).item() * len(labels)
                val_correct += (logits.argmax(1) == labels).sum().item()
                count += len(labels)
        if count != len(validation_data) or state['samples'] != len(train_data):
            raise RuntimeError('Incomplete train/validation population')
        val_loss = val_loss_sum / count
        if not math.isfinite(val_loss):
            raise RuntimeError('Nonfinite validation loss')
        scheduler.step(val_loss)
        improved = val_loss < state['best_val_loss'] - args.min_delta
        if improved:
            state['best_val_loss'] = val_loss
            state['best_epoch'] = epoch + 1
            state['epochs_without_improvement'] = 0
        else:
            state['epochs_without_improvement'] += 1
        metrics = dict(
            epoch=epoch+1, train_loss=state['loss_sum']/state['samples'],
            train_accuracy=state['correct']/state['samples'],
            validation_loss=val_loss, validation_accuracy=val_correct/count,
            lr=optimizer.param_groups[0]['lr'], improved=improved,
            best_epoch=state['best_epoch'], best_val_loss=state['best_val_loss'],
            epochs_without_improvement=state['epochs_without_improvement'],
            elapsed_seconds=round(time.monotonic()-started, 1),
        )
        state['history'].append(metrics)
        state.update(epoch=epoch+1, next_batch=0, loss_sum=0., correct=0, samples=0)
        model.train()  # save the train-mode ESCNN state, consistent with restore
        if improved:
            save(best)
        save(last)
        print(json.dumps(metrics, allow_nan=False), flush=True)

    stopped = state['epochs_without_improvement'] >= args.patience
    print(f"Training complete/stopped: epochs={state['epoch']} "
          f"reason={'early_stopping' if stopped else 'max_epochs'} "
          f"best_epoch={state['best_epoch']} best_val_loss={state['best_val_loss']:.6f} "
          f"best={best} last={last}", flush=True)


if __name__ == '__main__':
    main()
