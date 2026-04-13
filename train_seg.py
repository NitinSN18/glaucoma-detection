import argparse
import os
import random
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from seg_model import build_seg_model


IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def first_existing(paths: Sequence[str]) -> Optional[str]:
    for p in paths:
        if Path(p).exists():
            return p
    return None


def positive_int(value: str) -> int:
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError("Value must be > 0")
    return ivalue


def non_negative_int(value: str) -> int:
    ivalue = int(value)
    if ivalue < 0:
        raise argparse.ArgumentTypeError("Value must be >= 0")
    return ivalue


def resolve_train_paths(images_dir: Optional[str], masks_dir: Optional[str]) -> Tuple[str, str]:
    if images_dir and masks_dir:
        return images_dir, masks_dir

    image_candidates = [
        "/kaggle/input/datasets/avinashreddy2309/glaucoma-detection/seg_data/seg_data/images",
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/REFUGE/train/Images",
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/REFUGE/train/Images_Cropped",
        "seg_data/images",
    ]
    mask_candidates = [
        "/kaggle/input/datasets/avinashreddy2309/glaucoma-detection/seg_data/seg_data/masks",
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/REFUGE/train/Masks",
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/REFUGE/train/Masks_Cropped",
        "seg_data/masks",
    ]

    img_dir = images_dir or first_existing(image_candidates)
    msk_dir = masks_dir or first_existing(mask_candidates)
    if not img_dir or not msk_dir:
        raise FileNotFoundError(
            "Could not locate segmentation dataset directories. "
            "Pass --images-dir and --masks-dir explicitly."
        )
    return img_dir, msk_dir


def find_mask_path(mask_dir: str, stem: str) -> Optional[Path]:
    base = Path(mask_dir)
    for ext in (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"):
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    return None


def mask_to_channels(mask_np: np.ndarray) -> np.ndarray:
    if mask_np.ndim == 3:
        mask_np = mask_np[..., 0]

    uniques = np.unique(mask_np)
    if 255 in uniques or 128 in uniques:
        disc = (mask_np == 255).astype(np.float32)
        cup = (mask_np == 128).astype(np.float32)
        return np.stack([disc, cup], axis=0)

    if 2 in uniques or 1 in uniques:
        cup = (mask_np == 2).astype(np.float32)
        disc = np.logical_or(mask_np == 1, mask_np == 2).astype(np.float32)
        return np.stack([disc, cup], axis=0)

    fg = (mask_np > 0).astype(np.float32)
    return np.stack([fg, np.zeros_like(fg)], axis=0)


class SegDataset(Dataset):
    def __init__(self, images_dir: str, masks_dir: str, img_size: int = 256, augment: bool = True):
        self.images_dir = Path(images_dir)
        self.masks_dir = Path(masks_dir)
        self.img_size = img_size
        self.augment = augment

        names: List[str] = []
        for p in sorted(self.images_dir.iterdir()):
            if p.is_file() and p.suffix.lower() in IMG_EXTS:
                if find_mask_path(str(self.masks_dir), p.stem) is not None:
                    names.append(p.name)

        if not names:
            raise RuntimeError("No image/mask pairs were found in the selected directories.")

        self.image_names = names
        self.normalize = transforms.Normalize([0.5] * 3, [0.5] * 3)

    def __len__(self) -> int:
        return len(self.image_names)

    def __getitem__(self, idx: int):
        image_name = self.image_names[idx]
        image_path = self.images_dir / image_name
        mask_path = find_mask_path(str(self.masks_dir), Path(image_name).stem)
        if mask_path is None:
            raise FileNotFoundError(f"Missing mask for image: {image_name}")

        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)

        image = image.resize((self.img_size, self.img_size), Image.BILINEAR)
        mask = mask.resize((self.img_size, self.img_size), Image.NEAREST)

        if self.augment:
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_LEFT_RIGHT)
                mask = mask.transpose(Image.FLIP_LEFT_RIGHT)
            if random.random() > 0.5:
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                mask = mask.transpose(Image.FLIP_TOP_BOTTOM)
            angle = random.uniform(-15, 15)
            image = image.rotate(angle, resample=Image.BILINEAR, fillcolor=0)
            mask = mask.rotate(angle, resample=Image.NEAREST, fillcolor=0)

        image_t = transforms.ToTensor()(image)
        image_t = self.normalize(image_t)

        mask_np = np.array(mask)
        mask_t = torch.tensor(mask_to_channels(mask_np), dtype=torch.float32)
        return image_t, mask_t


class SegLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    @staticmethod
    def dice_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        pred = torch.sigmoid(pred)
        inter = (pred * target).sum(dim=(0, 2, 3))
        union = pred.sum(dim=(0, 2, 3)) + target.sum(dim=(0, 2, 3))
        return 1.0 - ((2.0 * inter + eps) / (union + eps)).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return self.bce(pred, target) + self.dice_loss(pred, target)


def train(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = detect_device()
    images_dir, masks_dir = resolve_train_paths(args.images_dir, args.masks_dir)

    save_path = args.save_path
    plot_path = args.plot_path
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    Path(plot_path).parent.mkdir(parents=True, exist_ok=True)

    print(f"Device: {device}")
    print(f"Images: {images_dir}")
    print(f"Masks : {masks_dir}")

    dataset = SegDataset(images_dir, masks_dir, img_size=args.img_size, augment=not args.no_augment)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = build_seg_model(num_classes=2, pretrained_backbone=args.pretrained_backbone).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    criterion = SegLoss()

    history: List[float] = []
    model.train()
    for epoch in range(args.epochs):
        epoch_loss = 0.0
        for images, masks in loader:
            images = images.to(device)
            masks = masks.to(device)

            pred = model(images)
            loss = criterion(pred, masks)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

            epoch_loss += float(loss.item())

        scheduler.step()
        avg = epoch_loss / max(1, len(loader))
        history.append(avg)
        print(f"Epoch {epoch + 1:02d}/{args.epochs} | loss={avg:.4f} | lr={scheduler.get_last_lr()[0]:.2e}")

    torch.save(model.state_dict(), save_path)
    print(f"Saved model: {save_path}")

    plt.figure(figsize=(8, 4))
    plt.plot(range(1, args.epochs + 1), history, marker="o", linewidth=2)
    plt.title("Segmentation Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path, dpi=120)
    print(f"Saved loss plot: {plot_path}")


def parse_args() -> argparse.Namespace:
    default_kaggle = Path("/kaggle/input").exists()
    default_save = "/kaggle/working/seg_model.pth" if default_kaggle else "seg_model.pth"
    default_plot = "/kaggle/working/seg_loss.png" if default_kaggle else "seg_loss.png"

    parser = argparse.ArgumentParser(description="Train DeepLab segmentation model for glaucoma datasets")
    parser.add_argument("--images-dir", type=str, default=None, help="Path to segmentation images")
    parser.add_argument("--masks-dir", type=str, default=None, help="Path to segmentation masks")
    parser.add_argument("--save-path", type=str, default=default_save, help="Model output file path")
    parser.add_argument("--plot-path", type=str, default=default_plot, help="Loss plot output path")
    parser.add_argument("--epochs", type=positive_int, default=25)
    parser.add_argument("--batch-size", type=positive_int, default=8)
    parser.add_argument("--img-size", type=positive_int, default=256)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--num-workers", type=non_negative_int, default=2 if default_kaggle else 0)
    parser.add_argument("--no-augment", action="store_true")
    parser.add_argument(
        "--pretrained-backbone",
        action="store_true",
        help="Enable ImageNet pretrained backbone weights (may require internet/download).",
    )
    return parser.parse_args()


if __name__ == "__main__":
    train(parse_args())
