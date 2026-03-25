import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

from seg_model import UNet

# ---- LOAD MODEL ----
model = UNet()
model.load_state_dict(torch.load("seg_model.pth"))
model.eval()

# ---- DEVICE ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)

# ---- TRANSFORM ----
transform = transforms.Compose([
    transforms.Resize((256,256)),
    transforms.ToTensor()
])

# ---- LOAD IMAGE ----
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw() # Hide the main window

image_path = filedialog.askopenfilename(
    title="Select a Fundus Image for Segmentation",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if not image_path:
    print("No image selected! Exiting.")
    exit()

image = Image.open(image_path).convert("RGB")

# ---- PREPROCESS ----
x = transform(image).unsqueeze(0).to(device)

# ---- PREDICT ----
with torch.no_grad():
    mask = model(x)
    mask = torch.sigmoid(mask)
    mask = (mask > 0.5).float()

# ---- MOVE TO CPU & RESIZE FOR PLOTTING ----
mask = mask.squeeze(0).cpu().numpy()  # shape: (2, 256, 256)

# Resize masks back to original image size
from PIL import Image as PILImage
disc_mask = np.array(PILImage.fromarray(mask[0]).resize((image.width, image.height)))
cup_mask  = np.array(PILImage.fromarray(mask[1]).resize((image.width, image.height)))

# ---- VISUALIZE ----
plt.figure(figsize=(15, 5))

# Original
plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# Disc overlay
plt.subplot(1, 3, 2)
plt.imshow(image)
plt.imshow(disc_mask, alpha=0.4, cmap="Blues")
plt.title("Optic Disc Mask")
plt.axis("off")

# Cup overlay
plt.subplot(1, 3, 3)
plt.imshow(image)
plt.imshow(cup_mask, alpha=0.4, cmap="Reds")
plt.title("Optic Cup Mask")
plt.axis("off")

plt.tight_layout()
plt.show()
