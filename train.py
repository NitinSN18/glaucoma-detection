#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'efficientnet_pytorch'. Install it with: pip install efficientnet_pytorch"
    ) from exc


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def default_data_paths() -> tuple[Path, Path]:
    if Path("/kaggle/input").exists():
        base = Path("/kaggle/input/datasets/avinashreddy2309/glaucoma-detection/data/data")
        return base / "train", base / "val"
    return Path("data/train"), Path("data/val")


def make_transforms() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ColorJitter(brightness=0.3, contrast=0.3),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )


def build_model(device: torch.device) -> nn.Module:
    model = EfficientNet.from_pretrained("efficientnet-b4")
    model._fc = nn.Linear(model._fc.in_features, 2)
    model = model.to(device)
    return model


def parse_args() -> argparse.Namespace:
    train_default, val_default = default_data_paths()

    parser = argparse.ArgumentParser(description="Train EfficientNet-B4 for glaucoma classification")
    parser.add_argument("--train-dir", default=str(train_default), help="Training data directory")
    parser.add_argument("--val-dir", default=str(val_default), help="Validation data directory")
    parser.add_argument("--epochs", type=int, default=10, help="Number of training epochs")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument(
        "--output",
        default="best_model.pth",
        help="Path to save best model checkpoint",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    train_path = Path(args.train_dir).expanduser().resolve()
    val_path = Path(args.val_dir).expanduser().resolve()

    print(f"Train path: {train_path}")
    print(f"Val path:   {val_path}")

    if not train_path.exists() or not val_path.exists():
        raise FileNotFoundError("Dataset path incorrect. Check --train-dir and --val-dir.")

    device = get_device()
    print(f"Using device: {device}")

    transform = make_transforms()
    train_data = datasets.ImageFolder(str(train_path), transform=transform)
    val_data = datasets.ImageFolder(str(val_path), transform=transform)

    print(f"Training images: {len(train_data)}")
    print(f"Validation images: {len(val_data)}")
    print(f"Classes: {train_data.classes}")

    num_workers = 0
    train_loader = DataLoader(
        train_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=num_workers,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=num_workers,
    )

    model = build_model(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    output_path = Path(args.output).expanduser().resolve()

    print("Starting training...")
    for epoch in range(args.epochs):
        print(f"\n--- Epoch {epoch + 1}/{args.epochs} ---")

        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        train_loss = running_loss / max(1, len(train_loader))
        print(f"Training Loss: {train_loss:.4f}")

        model.eval()
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device)

                outputs = model(images)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        accuracy = 100.0 * correct / max(1, total)
        print(f"Validation Accuracy: {accuracy:.2f}%")

        if accuracy > best_acc:
            best_acc = accuracy
            output_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), str(output_path))
            print(f"Saved best model to: {output_path}")

    print("Training complete")
    print(f"Best validation accuracy: {best_acc:.2f}%")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
