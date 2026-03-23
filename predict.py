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

# ---- IMAGE PATH (cross-platform) ----
# Priority:
#   1. Command-line argument:  python predict.py path/to/image.jpg
#   2. Auto-search inside  data/train  and  data/val  (any .jpg/.png)
#   3. Any .jpg / .png sitting next to this script

def find_test_image() -> Path:
    """Return the path of a test image, searching common project locations."""
    # 1. CLI argument
    if len(sys.argv) > 1:
        p = Path(sys.argv[1])
        if p.exists():
            return p
        raise FileNotFoundError(f"Image specified on command line not found: {p}")

    script_dir = Path(__file__).resolve().parent

    # 2. Search inside data/train and data/val sub-folders
    search_roots = [
        #script_dir / "data" / "train",
        script_dir / "data" / "test",

    ]
    for root in search_roots:
        if root.exists():
            for ext in ("*.jpg", "*.jpeg", "*.png"):
                matches = list(root.rglob(ext))
                if matches:
                    print(f"Auto-found test image: {matches[0]}")
                    return matches[0]

    # 3. Fallback: any image sitting next to the script
    for ext in ("*.jpg", "*.jpeg", "*.png"):
        matches = list(script_dir.glob(ext))
        if matches:
            print(f"Auto-found test image: {matches[0]}")
            return matches[0]

    raise FileNotFoundError(
        "No test image found.\n"
        "  • Pass the image path as an argument:  python predict.py path/to/image.jpg\n"
        "  • Or place an image inside data/train, data/val, or the project folder."
    )

image_path = find_test_image()

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
