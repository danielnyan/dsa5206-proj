"""Basic laptop training: notebook model, frozen VAL1 split, resumable Adam."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time


def atomic_save(state, path):
    """Keep the last complete checkpoint until its replacement is durable."""
    import torch
    temporary = path.with_suffix('.tmp')
    with temporary.open('wb') as stream:
        torch.save(state, stream)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True,
                        help='Extracted root containing val_dataset_1_norm and val_dataset_1_tu')
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30, help='Total target epochs, including resumed epochs')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--checkpoint-every', type=int, default=100, help='Save every N training batches')
    parser.add_argument('--resume', type=Path, help='Trusted last.pt checkpoint from this script')
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    args = parser.parse_args(argv)
    if min(args.epochs, args.batch_size, args.checkpoint_every) < 1:
        parser.error('Epochs, batch size and checkpoint interval must be positive')
    return args


def main(argv=None):
    args = parse_args(argv)
    import torch
    import escnn
    import numpy as np
    import random
    from PIL import Image
    from torch.utils.data import DataLoader, Dataset, Subset
    from torchvision.transforms import functional as TF, InterpolationMode
    from . import reimplementation as r
    from .group_equivariant_model import D8MultiScaleResNetGAP

    repository = Path(__file__).resolve().parents[1]
    r.ensure_output_separation(args.data, [args.output])
    last = args.output / 'last.pt'
    if last.exists() and args.resume is None:
        raise ValueError('Output already contains last.pt; use --resume or a new output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    train, validation, split = r.freeze_split(repository / 'manifests/VAL1_manifest.csv',
                                             args.output / 'split')

    class Patches(Dataset):
        def __init__(self, rows):
            self.paths = [r.image_path(row, args.data) for row in rows]
            self.labels = [row['ground_truth_code'] for row in rows]
            missing = next((p for p in self.paths if not p.is_file()), None)
            if missing is not None:
                raise FileNotFoundError(f'Missing image: {missing}; check --data')

        def __len__(self):
            return len(self.paths)

        def __getitem__(self, index):
            with Image.open(self.paths[index]) as image:
                image = TF.pil_to_tensor(image.convert('RGB'))
            image = TF.resize(image, [350, 350], interpolation=InterpolationMode.BILINEAR,
                              antialias=True)
            return image.float() / 127.5 - 1.0, self.labels[index]

    train_data, validation_data = Patches(train), Patches(validation)
    device = torch.device('cuda' if args.device == 'auto' and torch.cuda.is_available()
                          else 'cpu' if args.device == 'auto' else args.device)
    random.seed(42)
    np.random.seed(42)
    torch.manual_seed(42)
    if device.type == 'cuda':
        torch.cuda.manual_seed_all(42)
    model = D8MultiScaleResNetGAP().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, patience=2)
    criterion = torch.nn.CrossEntropyLoss()
    contract = dict(split=split['fingerprints'], batch_size=args.batch_size, seed=42,
                    model_sha256=hashlib.sha256((repository / 'modern_pca/group_equivariant_model.py').read_bytes()).hexdigest(),
                    torch=torch.__version__, escnn=escnn.__version__, preprocessing='RGB-bilinear350-minus1-plus1')
    state = dict(epoch=0, next_batch=0, loss_sum=0., correct=0, samples=0, history=[])
    if args.resume:
        saved = torch.load(args.resume, map_location='cpu', weights_only=False)
        if saved['contract'] != contract:
            raise ValueError('Checkpoint split/model/batch size/environment differs; use the original configuration')
        model.load_state_dict(saved['model'])
        optimizer.load_state_dict(saved['optimizer'])
        scheduler.load_state_dict(saved['scheduler'])
        state = saved['progress']
        random.setstate(saved['python_rng'])
        np.random.set_state(saved['numpy_rng'])
        torch.set_rng_state(saved['rng'])
        if device.type == 'cuda' and saved['cuda_rng'] is not None:
            torch.cuda.set_rng_state_all(saved['cuda_rng'])
        print(f"Resuming epoch {state['epoch'] + 1}, batch {state['next_batch']}", flush=True)

    def save():
        atomic_save(dict(contract=contract, progress=state, model=model.state_dict(),
                         optimizer=optimizer.state_dict(), scheduler=scheduler.state_dict(),
                         rng=torch.get_rng_state(), python_rng=random.getstate(),
                         numpy_rng=np.random.get_state(),
                         cuda_rng=torch.cuda.get_rng_state_all() if device.type == 'cuda' else None), last)

    print(f'{device}: {len(train_data)} training / {len(validation_data)} validation; all parameters trainable', flush=True)
    save()  # Also preserves the random initialization before the first batch.
    valid_loader = DataLoader(validation_data, batch_size=args.batch_size, num_workers=0)
    while state['epoch'] < args.epochs:
        epoch = state['epoch']
        order = torch.randperm(len(train_data), generator=torch.Generator().manual_seed(42 + epoch)).tolist()
        offset = state['next_batch'] * args.batch_size
        # No random augmentations or workers: restarting does not change the remaining samples.
        loader = DataLoader(Subset(train_data, order[offset:]), batch_size=args.batch_size,
                            num_workers=0, generator=torch.Generator().manual_seed(42 + epoch))
        model.train()
        saved_at = time.monotonic()
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(images)
            loss = criterion(logits, labels)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite loss; resume from last.pt after fixing the cause')
            loss.backward()
            optimizer.step()
            state['next_batch'] += 1
            state['loss_sum'] += loss.item() * len(labels)
            state['correct'] += (logits.argmax(1) == labels).sum().item()
            state['samples'] += len(labels)
            if state['next_batch'] % args.checkpoint_every == 0 or time.monotonic() - saved_at >= 300:
                save()
                saved_at = time.monotonic()
                print(f"Epoch {epoch + 1}/{args.epochs}, samples {state['samples']}/{len(train_data)}; checkpoint saved", flush=True)
        save()  # Validation interruption resumes here without repeating training updates.
        model.eval()
        val_loss, val_correct, count = 0., 0, 0
        with torch.no_grad():
            for images, labels in valid_loader:
                images, labels = images.to(device), labels.to(device)
                logits = model(images)
                val_loss += criterion(logits, labels).item() * len(labels)
                val_correct += (logits.argmax(1) == labels).sum().item()
                count += len(labels)
        val_loss /= count
        if not torch.isfinite(torch.tensor(val_loss)):
            raise RuntimeError('Nonfinite validation loss')
        scheduler.step(val_loss)
        metrics = dict(epoch=epoch + 1, train_loss=state['loss_sum'] / state['samples'],
                       train_accuracy=state['correct'] / state['samples'],
                       validation_loss=val_loss, validation_accuracy=val_correct / count)
        state['history'].append(metrics)
        state.update(epoch=epoch + 1, next_batch=0, loss_sum=0., correct=0, samples=0)
        model.train()  # escnn train/eval changes derived buffers; checkpoints use training mode.
        save()
        print(json.dumps(metrics), flush=True)
    print(f"Completed {args.epochs} epochs. Model and training state: {last}", flush=True)


if __name__ == '__main__':
    main()
