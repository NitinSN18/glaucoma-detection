import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet

# ---- IMAGE PATH (Mac style) ----
image_path = "t2.jpg"   # put your test image in project folder

# ---- DEVICE ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# ---- TRANSFORM ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---- LOAD MODEL ----
model = EfficientNet.from_pretrained('efficientnet-b0')  # match training
model._fc = nn.Linear(model._fc.in_features, 2)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
model.eval()

# ---- LOAD IMAGE ----
image = Image.open(image_path).convert("RGB")
image = transform(image).unsqueeze(0).to(device)

# ---- PREDICT ----
with torch.no_grad():
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
    confidence, pred = torch.max(probs, 1)

classes = ["glaucoma", "normal"]

print("Prediction:", classes[pred.item()])
print("Confidence:", confidence.item())

