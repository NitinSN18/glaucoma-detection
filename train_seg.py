import os
import random
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.transforms import ColorJitter
from torchvision.transforms import functional as TF
from torchvision.transforms.functional import InterpolationMode
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from seg_model import create_segmentation_model


SEED = 42
IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 50
VAL_RATIO = 0.1

FULL_FUNDUS_DIR = os.getenv("FULL_FUNDUS_DIR", "/Users/avinash/Downloads/full-fundus")
OPTIC_CUP_DIR = os.getenv("OPTIC_CUP_DIR", "/Users/avinash/Downloads/optic-cup")
OPTIC_DISC_DIR = os.getenv("OPTIC_DISC_DIR", "/Users/avinash/Downloads/optic-disc")

SEG_MODEL_ARCH = os.getenv("SEG_MODEL_ARCH", "deeplabv3plus").strip().lower()
SEG_MODEL_ENCODER = os.getenv("SEG_MODEL_ENCODER", "resnet34")
SEG_MODEL_PATH = os.getenv(
    "SEG_MODEL_PATH",
    "seg_model.pth" if SEG_MODEL_ARCH == "unet" else f"seg_model_{SEG_MODEL_ARCH}.pth",
)

print("\n[SEG-TRAIN DEBUG] Startup configuration", flush=True)
print(f"[SEG-TRAIN DEBUG] Script: {os.path.abspath(__file__)}", flush=True)
print(f"[SEG-TRAIN DEBUG] Model architecture: {SEG_MODEL_ARCH}", flush=True)
print(f"[SEG-TRAIN DEBUG] Backbone encoder: {SEG_MODEL_ENCODER}", flush=True)
print(f"[SEG-TRAIN DEBUG] Checkpoint path: {SEG_MODEL_PATH}\n", flush=True)


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SegDataset(Dataset):
    def __init__(self, samples, augment=False):
        self.samples = samples
        self.augment = augment

        self.color_jitter = ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.03,
        )

    def __len__(self):
        return len(self.samples)

    def _joint_augment(self, image, disc_mask, cup_mask):
        # Simulate full-frame fundus by random zoom-in crop (removes black border in many samples).
        if random.random() < 0.45:
            img_w, img_h = image.size
            scale = random.uniform(0.65, 0.95)
            crop_w = max(32, int(img_w * scale))
            crop_h = max(32, int(img_h * scale))

            left = random.randint(0, max(0, img_w - crop_w))
            top = random.randint(0, max(0, img_h - crop_h))

            image = TF.crop(image, top, left, crop_h, crop_w)
            disc_mask = TF.crop(disc_mask, top, left, crop_h, crop_w)
            cup_mask = TF.crop(cup_mask, top, left, crop_h, crop_w)

        if random.random() < 0.5:
            image = TF.hflip(image)
            disc_mask = TF.hflip(disc_mask)
            cup_mask = TF.hflip(cup_mask)

        if random.random() < 0.2:
            image = TF.vflip(image)
            disc_mask = TF.vflip(disc_mask)
            cup_mask = TF.vflip(cup_mask)

        angle = random.uniform(-18, 18)
        image = TF.rotate(
            image,
            angle=angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        disc_mask = TF.rotate(
            disc_mask,
            angle=angle,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        cup_mask = TF.rotate(
            cup_mask,
            angle=angle,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )

        max_shift = int(0.05 * min(image.size))
        tx = random.randint(-max_shift, max_shift)
        ty = random.randint(-max_shift, max_shift)
        scale = random.uniform(0.95, 1.05)

        image = TF.affine(
            image,
            angle=0,
            translate=(tx, ty),
            scale=scale,
            shear=0,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        disc_mask = TF.affine(
            disc_mask,
            angle=0,
            translate=(tx, ty),
            scale=scale,
            shear=0,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )
        cup_mask = TF.affine(
            cup_mask,
            angle=0,
            translate=(tx, ty),
            scale=scale,
            shear=0,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )

        image = self.color_jitter(image)
        return image, disc_mask, cup_mask

    def __getitem__(self, idx):
        sample = self.samples[idx]
        img_path = sample["image"]
        cup_path = sample["cup"]
        disc_path = sample["disc"]
        sample_name = sample["name"]

        if not os.path.exists(img_path) or not os.path.exists(cup_path) or not os.path.exists(disc_path):
            raise FileNotFoundError(f"Missing file in sample {sample_name}")

        image = Image.open(img_path).convert("RGB")
        cup_mask = Image.open(cup_path).convert("L")
        disc_mask = Image.open(disc_path).convert("L")

        if image.size != cup_mask.size or image.size != disc_mask.size:
            raise ValueError(
                f"Size mismatch in {sample_name}: image={image.size}, cup={cup_mask.size}, disc={disc_mask.size}"
            )

        if self.augment:
            image, disc_mask, cup_mask = self._joint_augment(image, disc_mask, cup_mask)

        image = TF.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        disc_mask = TF.resize(
            disc_mask,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.NEAREST,
        )
        cup_mask = TF.resize(
            cup_mask,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.NEAREST,
        )

        image_tensor = TF.to_tensor(image)

        disc_raw = np.array(disc_mask, dtype=np.float32)
        cup_raw = np.array(cup_mask, dtype=np.float32)

        disc_bin = (disc_raw > 0).astype(np.float32)
        cup_bin = (cup_raw > 0).astype(np.float32)

        mask_2ch = np.stack([disc_bin, cup_bin], axis=0)
        mask_tensor = torch.from_numpy(mask_2ch)

        return image_tensor, mask_tensor


def _list_images(folder_path):
    return sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith(".")
    ])


def _to_name_map(folder_path):
    name_map = {}
    for filename in _list_images(folder_path):
        basename = os.path.splitext(filename)[0]
        name_map[basename] = os.path.join(folder_path, filename)
    return name_map


def build_triplet_samples(image_dir, cup_dir, disc_dir):
    image_map = _to_name_map(image_dir)
    cup_map = _to_name_map(cup_dir)
    disc_map = _to_name_map(disc_dir)

    image_names = set(image_map.keys())
    cup_names = set(cup_map.keys())
    disc_names = set(disc_map.keys())

    matched = sorted(image_names & cup_names & disc_names)
    missing_cup = sorted(image_names - cup_names)
    missing_disc = sorted(image_names - disc_names)

    samples = [
        {
            "name": name,
            "image": image_map[name],
            "cup": cup_map[name],
            "disc": disc_map[name],
        }
        for name in matched
    ]

    report = {
        "image_total": len(image_names),
        "cup_total": len(cup_names),
        "disc_total": len(disc_names),
        "matched_total": len(matched),
        "missing_cup_total": len(missing_cup),
        "missing_disc_total": len(missing_disc),
    }

    return samples, report


class DiceBCELoss(nn.Module):
    def __init__(self):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1.0):
        bce_loss = self.bce(inputs, targets)
        probs = torch.sigmoid(inputs)

        probs_flat = probs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (probs_flat * targets_flat).sum()
        dice_loss = 1 - (2.0 * intersection + smooth) / (
            probs_flat.sum() + targets_flat.sum() + smooth
        )

        return bce_loss + dice_loss


def batch_metrics_from_logits(logits, targets, threshold=0.5, eps=1e-6):
    probs = torch.sigmoid(logits)
    preds = (probs >= threshold).float()

    pixel_acc = (preds == targets).float().mean().item()

    disc_acc = (preds[:, 0] == targets[:, 0]).float().mean().item()
    cup_acc = (preds[:, 1] == targets[:, 1]).float().mean().item()

    disc_inter = (preds[:, 0] * targets[:, 0]).sum()
    disc_union = preds[:, 0].sum() + targets[:, 0].sum()
    disc_dice = ((2.0 * disc_inter + eps) / (disc_union + eps)).item()

    cup_inter = (preds[:, 1] * targets[:, 1]).sum()
    cup_union = preds[:, 1].sum() + targets[:, 1].sum()
    cup_dice = ((2.0 * cup_inter + eps) / (cup_union + eps)).item()

    return pixel_acc, disc_acc, cup_acc, disc_dice, cup_dice


set_seed()

# ---- DATA ----
all_samples, pairing_report = build_triplet_samples(
    FULL_FUNDUS_DIR,
    OPTIC_CUP_DIR,
    OPTIC_DISC_DIR,
)

print("\nTriplet pairing report")
print(f"  Full-fundus images : {pairing_report['image_total']}")
print(f"  Optic-cup masks    : {pairing_report['cup_total']}")
print(f"  Optic-disc masks   : {pairing_report['disc_total']}")
print(f"  Matched triplets   : {pairing_report['matched_total']}")
print(f"  Missing cup masks  : {pairing_report['missing_cup_total']}")
print(f"  Missing disc masks : {pairing_report['missing_disc_total']}")

if len(all_samples) < 10:
    raise ValueError("Need at least 10 images to create a stable train/val split.")

random.shuffle(all_samples)
val_count = max(1, int(len(all_samples) * VAL_RATIO))
val_samples = all_samples[:val_count]
train_samples = all_samples[val_count:]

train_dataset = SegDataset(train_samples, augment=True)
val_dataset = SegDataset(val_samples, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

# ---- MODEL ----
model = create_segmentation_model(
    arch=SEG_MODEL_ARCH,
    out_channels=2,
    encoder_name=SEG_MODEL_ENCODER,
    encoder_weights="imagenet",
)
print(f"Model architecture: {SEG_MODEL_ARCH}")
if SEG_MODEL_ARCH in {"deeplabv3+", "deeplabv3plus"}:
    print(f"Backbone encoder: {SEG_MODEL_ENCODER}")

if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple M1 GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

model.to(device)

criterion = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

train_losses = []
val_losses = []
best_val = float("inf")
patience = 8
no_improve = 0

# ---- TRAINING ----
for epoch in range(EPOCHS):
    model.train()
    train_total = 0.0
    train_batches = 0
    train_acc_total = 0.0
    train_disc_acc_total = 0.0
    train_cup_acc_total = 0.0
    train_disc_dice_total = 0.0
    train_cup_dice_total = 0.0

    for images, masks in train_loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_total += loss.item()
        train_batches += 1

        with torch.no_grad():
            m = batch_metrics_from_logits(outputs.detach(), masks)
            train_acc_total += m[0]
            train_disc_acc_total += m[1]
            train_cup_acc_total += m[2]
            train_disc_dice_total += m[3]
            train_cup_dice_total += m[4]

    avg_train_loss = train_total / max(1, train_batches)
    avg_train_acc = train_acc_total / max(1, train_batches)
    avg_train_disc_acc = train_disc_acc_total / max(1, train_batches)
    avg_train_cup_acc = train_cup_acc_total / max(1, train_batches)
    avg_train_disc_dice = train_disc_dice_total / max(1, train_batches)
    avg_train_cup_dice = train_cup_dice_total / max(1, train_batches)
    train_losses.append(avg_train_loss)

    model.eval()
    val_total = 0.0
    val_batches = 0
    val_acc_total = 0.0
    val_disc_acc_total = 0.0
    val_cup_acc_total = 0.0
    val_disc_dice_total = 0.0
    val_cup_dice_total = 0.0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_total += loss.item()
            val_batches += 1

            m = batch_metrics_from_logits(outputs, masks)
            val_acc_total += m[0]
            val_disc_acc_total += m[1]
            val_cup_acc_total += m[2]
            val_disc_dice_total += m[3]
            val_cup_dice_total += m[4]

    avg_val_loss = val_total / max(1, val_batches)
    avg_val_acc = val_acc_total / max(1, val_batches)
    avg_val_disc_acc = val_disc_acc_total / max(1, val_batches)
    avg_val_cup_acc = val_cup_acc_total / max(1, val_batches)
    avg_val_disc_dice = val_disc_dice_total / max(1, val_batches)
    avg_val_cup_dice = val_cup_dice_total / max(1, val_batches)
    val_losses.append(avg_val_loss)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.6f} | "
        f"Val Loss: {avg_val_loss:.6f} | "
        f"Train Acc: {avg_train_acc:.4f} | "
        f"Val Acc: {avg_val_acc:.4f} | "
        f"Train Disc Acc: {avg_train_disc_acc:.4f} | "
        f"Train Cup Acc: {avg_train_cup_acc:.4f} | "
        f"Val Disc Acc: {avg_val_disc_acc:.4f} | "
        f"Val Cup Acc: {avg_val_cup_acc:.4f} | "
        f"Train Disc Dice: {avg_train_disc_dice:.4f} | "
        f"Train Cup Dice: {avg_train_cup_dice:.4f} | "
        f"Val Disc Dice: {avg_val_disc_dice:.4f} | "
        f"Val Cup Dice: {avg_val_cup_dice:.4f}"
    )

    if avg_val_loss < best_val:
        best_val = avg_val_loss
        no_improve = 0
        torch.save(model.state_dict(), SEG_MODEL_PATH)
        print(f"  Saved best model -> {SEG_MODEL_PATH}")
    else:
        no_improve += 1
        if no_improve >= patience:
            print("Early stopping triggered.")
            break

print(f"\nTraining complete. Best Val Loss: {best_val:.6f}")

plt.figure(figsize=(8, 5))
plt.plot(train_losses, label="Train Loss")
plt.plot(val_losses, label="Val Loss")
plt.title("Segmentation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.tight_layout()
plt.show()
