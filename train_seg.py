import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

from seg_model import UNet

class SegDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir

        # STRICT FILTER (fixes EVERYTHING)
        self.images = [
            f for f in os.listdir(image_dir)
            if f.lower().endswith((".jpg", ".jpeg", ".png"))
            and not f.startswith(".")
        ]

        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_name = self.images[idx]
        img_path = os.path.join(self.image_dir, img_name)

        # Remove extension
        name = os.path.splitext(img_name)[0]

        # Mask assumed to be PNG (change if needed)
        mask_path = os.path.join(self.mask_dir, name + ".png")

        # ---- DEBUG CHECK ----
        if not os.path.exists(mask_path):
            raise FileNotFoundError(f"Mask not found for {img_name} → {mask_path}")

        image = Image.open(img_path).convert("RGB")

        # Mask values are: 0 (background), 1 (disc), 2 (cup)
        raw_mask = np.array(Image.open(mask_path).convert("L"), dtype=np.float32)
        disc_mask = (raw_mask >= 1).astype(np.float32)   # disc includes both 1 and 2
        cup_mask  = (raw_mask == 2).astype(np.float32)   # cup is only 2

        # Stack → (2, H, W)
        mask_2ch = np.stack([disc_mask, cup_mask], axis=0)  # shape: (2, H, W)

        if self.transform:
            image = self.transform(image)
            # Resize mask channels individually to match image size
            import torch.nn.functional as F
            mask_tensor = torch.from_numpy(mask_2ch).unsqueeze(0)  # (1, 2, H, W)
            mask_tensor = F.interpolate(mask_tensor, size=(256, 256), mode="nearest")
            mask_2ch = mask_tensor.squeeze(0)  # (2, 256, 256)
        else:
            mask_2ch = torch.from_numpy(mask_2ch)

        return image, mask_2ch

class DiceBCELoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(DiceBCELoss, self).__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, inputs, targets, smooth=1.0):
        # Calculate BCE loss using raw logits
        bce_loss = self.bce(inputs, targets)

        # Apply sigmoid to inputs for Dice Loss
        inputs = torch.sigmoid(inputs)       

        # Flatten mask and prediction tensors
        inputs_flat = inputs.view(-1)
        targets_flat = targets.view(-1)

        intersection = (inputs_flat * targets_flat).sum()                            
        dice_loss = 1 - (2. * intersection + smooth) / (inputs_flat.sum() + targets_flat.sum() + smooth)  
        
        # Combine the losses
        return bce_loss + dice_loss


# ---- TRANSFORMS ----
# CenterCrop removed — cup/disc masks already define the target regions precisely.
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ---- DATA ----
dataset = SegDataset("seg_data/images", "seg_data/masks", transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Total images found: {len(dataset)}")

# ---- MODEL ----
# out_channels=2: channel 0 = optic disc, channel 1 = optic cup
model = UNet(out_channels=2)

# ---- DEVICE (M1 GPU SUPPORT) ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple M1 GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

model.to(device)

# ---- LOSS + OPTIMIZER ----
criterion = DiceBCELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

losses = []

# ---- TRAINING ----
epochs = 10

for epoch in range(epochs):
    total_loss = 0

    for images, masks in loader:
        images = images.to(device)
        masks = masks.to(device)

        outputs = model(images)
        loss = criterion(outputs, masks)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    losses.append(total_loss)

    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# ---- SAVE MODEL ----
torch.save(model.state_dict(), "seg_model.pth")
print("Model saved as seg_model.pth")

# ---- PLOT LOSS ----
plt.plot(losses)
plt.title("Segmentation Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.show()