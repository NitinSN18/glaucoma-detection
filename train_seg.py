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

from seg_model import UNet


SEED = 42
IMAGE_SIZE = 256
BATCH_SIZE = 4
EPOCHS = 50
VAL_RATIO = 0.1


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, image_names=None, augment=False):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.augment = augment

        if image_names is None:
            self.images = sorted([
                f for f in os.listdir(image_dir)
                if f.lower().endswith((".jpg", ".jpeg", ".png")) and not f.startswith(".")
            ])
        else:
            self.images = sorted(image_names)

        self.color_jitter = ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.15,
            hue=0.03,
        )

    def __len__(self):
        return len(self.images)

    def _joint_augment(self, image, mask):
        # Simulate full-frame fundus by random zoom-in crop (removes black border in many samples).
        if random.random() < 0.45:
            img_w, img_h = image.size
            scale = random.uniform(0.65, 0.95)
            crop_w = max(32, int(img_w * scale))
            crop_h = max(32, int(img_h * scale))

            left = random.randint(0, max(0, img_w - crop_w))
            top = random.randint(0, max(0, img_h - crop_h))

            image = TF.crop(image, top, left, crop_h, crop_w)
            mask = TF.crop(mask, top, left, crop_h, crop_w)

        if random.random() < 0.5:
            image = TF.hflip(image)
            mask = TF.hflip(mask)

        if random.random() < 0.2:
            image = TF.vflip(image)
            mask = TF.vflip(mask)

        angle = random.uniform(-18, 18)
        image = TF.rotate(
            image,
            angle=angle,
            interpolation=InterpolationMode.BILINEAR,
            fill=0,
        )
        mask = TF.rotate(
            mask,
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
        mask = TF.affine(
            mask,
            angle=0,
            translate=(tx, ty),
            scale=scale,
            shear=0,
            interpolation=InterpolationMode.NEAREST,
            fill=0,
        )

        image = self.color_jitter(image)
        return image, mask

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)
        name = os.path.splitext(img_name)[0]
        mask_path = os.path.join(self.mask_dir, name + ".png")

        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for {img_name} -> {mask_path}")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.augment:
            image, mask = self._joint_augment(image, mask)

        image = TF.resize(
            image,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.BILINEAR,
            antialias=True,
        )
        mask = TF.resize(
            mask,
            [IMAGE_SIZE, IMAGE_SIZE],
            interpolation=InterpolationMode.NEAREST,
        )

        image_tensor = TF.to_tensor(image)

        raw_mask = np.array(mask, dtype=np.float32)
        disc_mask = (raw_mask >= 1).astype(np.float32)
        cup_mask = (raw_mask == 2).astype(np.float32)
        mask_2ch = np.stack([disc_mask, cup_mask], axis=0)
        mask_tensor = torch.from_numpy(mask_2ch)

        return image_tensor, mask_tensor


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


set_seed()

# ---- DATA ----
base_dataset = SegDataset("seg_data/images", "seg_data/masks", augment=False)
all_names = base_dataset.images

if len(all_names) < 10:
    raise ValueError("Need at least 10 images to create a stable train/val split.")

random.shuffle(all_names)
val_count = max(1, int(len(all_names) * VAL_RATIO))
val_names = all_names[:val_count]
train_names = all_names[val_count:]

train_dataset = SegDataset("seg_data/images", "seg_data/masks", image_names=train_names, augment=True)
val_dataset = SegDataset("seg_data/images", "seg_data/masks", image_names=val_names, augment=False)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

print(f"Train images: {len(train_dataset)} | Val images: {len(val_dataset)}")

# ---- MODEL ----
model = UNet(out_channels=2)

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

    avg_train_loss = train_total / max(1, train_batches)
    train_losses.append(avg_train_loss)

    model.eval()
    val_total = 0.0
    val_batches = 0

    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            val_total += loss.item()
            val_batches += 1

    avg_val_loss = val_total / max(1, val_batches)
    val_losses.append(avg_val_loss)

    print(
        f"Epoch {epoch + 1:02d}/{EPOCHS} | "
        f"Train Loss: {avg_train_loss:.6f} | "
        f"Val Loss: {avg_val_loss:.6f}"
    )

    if avg_val_loss < best_val:
        best_val = avg_val_loss
        no_improve = 0
        torch.save(model.state_dict(), "seg_model.pth")
        print("  Saved best model -> seg_model.pth")
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
