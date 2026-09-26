"""Train MedMamba on ImageFolder-compatible train and validation directories."""

import argparse
import json
import os
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm

from MedMamba import create_model, load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--val-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--variant", choices=("tiny", "small", "base"), default="tiny")
    parser.add_argument("--weights", type=Path, help="Optional checkpoint to resume/fine-tune")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--workers", type=int, default=min(8, os.cpu_count() or 1))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--no-amp", action="store_true", help="Disable CUDA mixed precision")
    return parser.parse_args()


def make_loaders(args):
    normalize = transforms.Normalize((0.5,) * 3, (0.5,) * 3)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(args.image_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        normalize,
    ])
    train_dataset = datasets.ImageFolder(args.train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(args.val_dir, transform=val_transform)
    if train_dataset.class_to_idx != val_dataset.class_to_idx:
        raise ValueError("Training and validation directories must contain the same classes")
    if not train_dataset or not val_dataset:
        raise ValueError("Training and validation datasets must not be empty")

    loader_options = {
        "batch_size": args.batch_size,
        "num_workers": args.workers,
        "pin_memory": args.device.startswith("cuda"),
        "persistent_workers": args.workers > 0,
    }
    train_loader = DataLoader(train_dataset, shuffle=True, **loader_options)
    val_loader = DataLoader(val_dataset, shuffle=False, **loader_options)
    return train_dataset, train_loader, val_loader


def run_epoch(model, loader, criterion, device, optimizer=None, scaler=None):
    training = optimizer is not None
    model.train(training)
    total_loss = correct = count = 0
    context = torch.enable_grad if training else torch.inference_mode
    with context():
        for images, labels in tqdm(loader, desc="train" if training else "val"):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            if training:
                optimizer.zero_grad(set_to_none=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.float16,
                enabled=scaler is not None,
            ):
                logits = model(images)
                loss = criterion(logits, labels)
            if training:
                if scaler:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
            total_loss += loss.item() * labels.size(0)
            correct += (logits.argmax(dim=1) == labels).sum().item()
            count += labels.size(0)
    return total_loss / count, correct / count


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")

    dataset, train_loader, val_loader = make_loaders(args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "class_indices.json").write_text(
        json.dumps({index: name for name, index in dataset.class_to_idx.items()}, indent=2)
    )

    model = create_model(args.variant, num_classes=len(dataset.classes)).to(device)
    if args.weights:
        load_checkpoint(model, args.weights, map_location=device)
        print(f"Loaded checkpoint: {args.weights}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    use_amp = device.type == "cuda" and not args.no_amp
    scaler = torch.cuda.amp.GradScaler(enabled=True) if use_amp else None
    best_accuracy = -1.0

    for epoch in range(1, args.epochs + 1):
        train_loss, train_accuracy = run_epoch(
            model, train_loader, criterion, device, optimizer, scaler
        )
        val_loss, val_accuracy = run_epoch(model, val_loader, criterion, device)
        print(
            f"epoch={epoch}/{args.epochs} train_loss={train_loss:.4f} "
            f"train_acc={train_accuracy:.4f} val_loss={val_loss:.4f} "
            f"val_acc={val_accuracy:.4f}"
        )
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "classes": dataset.classes,
            "variant": args.variant,
        }
        torch.save(state, args.output_dir / "last.pt")
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            torch.save(state, args.output_dir / "best.pt")

    print(f"Training complete. Best validation accuracy: {best_accuracy:.4f}")


if __name__ == "__main__":
    main()
