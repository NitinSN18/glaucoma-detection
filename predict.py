import os
import sys
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet
from pathlib import Path

# ---- DEVICE (matches train.py: CUDA > MPS > CPU) ----
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# ---- IMAGE PATH ----
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw() # Hide the main window

image_path = filedialog.askopenfilename(
    title="Select a Fundus Image for Classification",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if not image_path:
    print("No image selected. Exiting.")
    sys.exit()

# ---- TRANSFORM ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---- LOAD MODEL (efficientnet-b0, matching train.py) ----
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, 2)

model_path = Path(__file__).resolve().parent / "best_model.pth"
if not model_path.exists():
    raise FileNotFoundError(f"Trained model not found at: {model_path}")

model.load_state_dict(torch.load(str(model_path), map_location=device))
model = model.to(device)
model.eval()

# ---- LOAD & PREDICT ----
image = Image.open(image_path).convert("RGB")
image_tensor = transform(image).unsqueeze(0).to(device)

with torch.no_grad():
    outputs = model(image_tensor)
    probs = torch.softmax(outputs, dim=1)
    confidence, pred = torch.max(probs, 1)

classes = ["glaucoma", "normal"]
print("Image     :", image_path)
print("Prediction:", classes[pred.item()])
print("Confidence:", f"{confidence.item():.4f}")
