import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

from seg_model import UNet

# ---- DEVICE ----
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using:", device)

# ---- PATHS ----
if Path("/kaggle/input").exists():
    base = "/kaggle/input/datasets/avinashreddy2309/glaucoma-detection/seg_data/seg_data"
    img_dir = os.path.join(base, "images")
    mask_dir = os.path.join(base, "masks")
    save_path = "/kaggle/working/seg_model.pth"
else:
    img_dir = "seg_data/images"
    mask_dir = "seg_data/masks"
    save_path = "seg_model.pth"

if not os.path.isdir(img_dir) or not os.path.isdir(mask_dir):
    raise FileNotFoundError(
        f"Segmentation data not found. Expected img_dir={img_dir}, mask_dir={mask_dir}."
    )

# ---- DATASET ----
# NOTE: Spatial augmentations (flip, rotation) are intentionally omitted here.
# Applying them only to the image without the same transform on the mask would
# produce misaligned image-mask pairs and corrupt training.
img_transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])


class SegDataset(Dataset):
    """Segmentation dataset that loads fundus images and 2-channel disc/cup masks.

    Mask pixel conventions (Magrabia/DRISHTI-style):
      - 255 → optic disc region (channel 0)
      - 128 → optic cup region  (channel 1)
    """

    def __init__(self, img_dir: str, mask_dir: str):
        self.imgs = sorted(
            f for f in os.listdir(img_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
        )
        self.img_dir = img_dir
        self.mask_dir = mask_dir

    def __len__(self) -> int:
        return len(self.imgs)

    def __getitem__(self, idx: int):
        name = self.imgs[idx]

        img = Image.open(os.path.join(self.img_dir, name)).convert("RGB")
        img = img_transform(img)

        mname = os.path.splitext(name)[0] + ".png"
        mpath = os.path.join(self.mask_dir, mname)

        mask = Image.open(mpath).convert("L")
        # Use NEAREST to avoid interpolation corrupting discrete label values
        mask = mask.resize((256, 256), resample=Image.NEAREST)
        mask_np = np.array(mask)

        disc = (mask_np == 255).astype(np.float32)
        cup = (mask_np == 128).astype(np.float32)

        return img, torch.tensor(np.stack([disc, cup], axis=0))


dataset = SegDataset(img_dir, mask_dir)
loader = DataLoader(dataset, batch_size=8, shuffle=True)
print("Samples:", len(dataset))

# ---- MODEL ----
model = UNet(out_channels=2).to(device)

# ---- LOSS ----
bce_loss_fn = nn.BCEWithLogitsLoss()


def dice_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    pred = torch.sigmoid(pred)
    inter = (pred * target).sum()
    return 1 - (2 * inter + 1) / (pred.sum() + target.sum() + 1)


def combined_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return bce_loss_fn(pred, target) + dice_loss(pred, target)


optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# ---- TRAINING ----
epochs = 25

for epoch in range(epochs):
    model.train()
    total_loss = 0.0

    for imgs, masks in loader:
        imgs = imgs.to(device)
        masks = masks.to(device)

        optimizer.zero_grad()
        out = model(imgs)
        loss = combined_loss(out, masks)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}/{epochs}: loss={total_loss / len(loader):.4f}")

# ---- SAVE ----
torch.save(model.state_dict(), save_path)
print(f"Saved seg_model.pth to {save_path}")
