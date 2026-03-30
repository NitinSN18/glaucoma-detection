import sys
import os
import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from pathlib import Path

try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError:
    print("Installing efficientnet_pytorch...")
    os.system("pip install efficientnet_pytorch")
    from efficientnet_pytorch import EfficientNet

# ---- DEVICE ----
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

print("Using device:", device)

# ---- TRANSFORM ----
# Must match the validation transform used in train.py
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---- MODEL ----
# Must use the same EfficientNet variant as train.py (efficientnet-b4)
model = EfficientNet.from_pretrained("efficientnet-b4")
model._fc = nn.Linear(model._fc.in_features, 2)

# ---- FIND MODEL WEIGHTS ----
possible_paths = []

kaggle_path = Path("/kaggle/working/best_model.pth")
if kaggle_path.exists():
    possible_paths.append(kaggle_path)

cwd_path = Path(os.getcwd()) / "best_model.pth"
if cwd_path.exists():
    possible_paths.append(cwd_path)

try:
    script_path = Path(__file__).resolve().parent / "best_model.pth"
    if script_path.exists():
        possible_paths.append(script_path)
except NameError:
    pass

if not possible_paths:
    raise FileNotFoundError(
        "best_model.pth not found. Run train.py first to train and save the model."
    )

model_path = possible_paths[0]
model.load_state_dict(torch.load(str(model_path), map_location=device))
model = model.to(device)
model.eval()

print("Loaded model from:", model_path)

# ---- COLLECT IMAGE PATHS ----
# ImageFolder sorts class names alphabetically: index 0 = glaucoma, index 1 = normal
CLASSES = ["glaucoma", "normal"]
VALID_EXT = (".jpg", ".jpeg", ".png")

image_paths = []

# CLI args (filter out non-image paths to avoid Jupyter kernel args)
if len(sys.argv) > 1:
    image_paths = [
        p for p in sys.argv[1:]
        if p.lower().endswith(VALID_EXT) and os.path.exists(p)
    ]

# Kaggle auto-discovery
if not image_paths and Path("/kaggle/input").exists():
    print("Running in Kaggle mode")
    base = "/kaggle/input/datasets/avinashreddy2309/glaucoma-detection/data/data/test"
    if os.path.isdir(base):
        image_paths = [
            os.path.join(base, f)
            for f in os.listdir(base)
            if f.lower().endswith(VALID_EXT)
        ]

# Local GUI fallback
if not image_paths:
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
        image_paths = [path]
    except Exception:
        print("No valid input. Usage: python predict.py <image_path>")
        sys.exit(1)

# ---- PREDICTION ----
for image_path in image_paths:
    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(image_tensor)
            probs = torch.softmax(outputs, dim=1)
            confidence, pred = torch.max(probs, 1)

        print(f"\nImage     : {image_path}")
        print(f"Prediction: {CLASSES[pred.item()]}")
        print(f"Confidence: {confidence.item():.4f}")

    except Exception as e:
        print(f"Error processing {image_path}: {e}")
