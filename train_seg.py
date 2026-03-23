import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
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
        mask = Image.open(mask_path).convert("L")

        if self.transform:
            image = self.transform(image)
            mask = self.transform(mask)

        return image, mask


# ---- TRANSFORMS ----
transform = transforms.Compose([
    transforms.CenterCrop(512),   # 🔥 removes black outer boundary
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

# ---- DATA ----
dataset = SegDataset("seg_data/images", "seg_data/masks", transform)
loader = DataLoader(dataset, batch_size=4, shuffle=True)

print(f"Total images found: {len(dataset)}")

# ---- MODEL ----
model = UNet()

# ---- DEVICE (M1 GPU SUPPORT) ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
    print("Using Apple M1 GPU (MPS)")
else:
    device = torch.device("cpu")
    print("Using CPU")

model.to(device)

# ---- LOSS + OPTIMIZER ----
criterion = nn.BCEWithLogitsLoss()
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