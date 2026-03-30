import sys
import os
import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
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

print("Using device:", device)

# ---- FIND MODEL WEIGHTS ----
possible_paths = []

kaggle_path = Path("/kaggle/working/seg_model.pth")
if kaggle_path.exists():
    possible_paths.append(kaggle_path)

cwd_path = Path(os.getcwd()) / "seg_model.pth"
if cwd_path.exists():
    possible_paths.append(cwd_path)

try:
    script_path = Path(__file__).resolve().parent / "seg_model.pth"
    if script_path.exists():
        possible_paths.append(script_path)
except NameError:
    pass

if not possible_paths:
    raise FileNotFoundError(
        "seg_model.pth not found. Run train_seg.py first to train and save the model."
    )

model_path = possible_paths[0]
model = UNet(out_channels=2)
model.load_state_dict(torch.load(str(model_path), map_location=device, weights_only=True))
model = model.to(device)
model.eval()
print("Loaded model from:", model_path)

# ---- TRANSFORM ----
# Must match the transform used in train_seg.py (no spatial augmentations)
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
])

# ---- COLLECT IMAGE PATH ----
VALID_EXT = (".jpg", ".jpeg", ".png")

img_path = None

# CLI arg
if len(sys.argv) > 1:
    candidate = sys.argv[1]
    if candidate.lower().endswith(VALID_EXT) and os.path.exists(candidate):
        img_path = candidate

# Kaggle default
if img_path is None and Path("/kaggle/input").exists():
    kaggle_img = (
        "/kaggle/input/datasets/avinashreddy2309/glaucoma-detection"
        "/data/data/test/t2.jpg"
    )
    if os.path.exists(kaggle_img):
        img_path = kaggle_img

# Local GUI fallback
if img_path is None:
    try:
        import tkinter as tk
        from tkinter import filedialog

        root = tk.Tk()
        root.withdraw()
        path = filedialog.askopenfilename(
            title="Select Fundus Image",
            filetypes=[("Image Files", "*.jpg *.jpeg *.png")],
        )
        if not path:
            print("No image selected.")
            sys.exit(0)
        img_path = path
    except Exception:
        print("No valid input. Usage: python predict_seg.py <image_path>")
        sys.exit(1)

# ---- PREDICT ----
img = Image.open(img_path).convert("RGB")
x = transform(img).unsqueeze(0).to(device)

with torch.no_grad():
    out = torch.sigmoid(model(x))
    out = (out > 0.3).float()

mask = out.squeeze().cpu().numpy()

disc = (mask[0] * 255).astype(np.uint8)
cup = (mask[1] * 255).astype(np.uint8)

disc_resized = np.array(Image.fromarray(disc).resize((img.width, img.height)))
cup_resized = np.array(Image.fromarray(cup).resize((img.width, img.height)))

# ---- DISPLAY ----
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(img)
plt.title("Original")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(img)
plt.imshow(disc_resized, alpha=0.4, cmap="Blues")
plt.title("Disc")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(img)
plt.imshow(cup_resized, alpha=0.4, cmap="Reds")
plt.title("Cup")
plt.axis("off")

plt.tight_layout()
plt.show()
