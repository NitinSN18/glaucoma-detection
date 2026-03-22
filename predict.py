import torch
import torch.nn as nn
from torchvision import transforms
from PIL import Image
from efficientnet_pytorch import EfficientNet

<<<<<<< HEAD
# ---- IMAGE PATH (Mac style) ----
image_path = "t2.jpg"   # put your test image in project folder

# ---- DEVICE ----
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
=======
# ---- CHANGE THIS ----
image_path = r"C:\Users\nitin\glaucoma-project\data\train\glaucoma\Im310_g_ACRIMA.jpg"
>>>>>>> 1c78d8cb5cc86a3abbae0f7274ab89f5b381c2a5

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
<<<<<<< HEAD
model = EfficientNet.from_pretrained('efficientnet-b0')  # match training
model._fc = nn.Linear(model._fc.in_features, 2)

model.load_state_dict(torch.load("best_model.pth", map_location=device))
model = model.to(device)
=======
model = EfficientNet.from_pretrained('efficientnet-b4')
model._fc = nn.Linear(model._fc.in_features, 2)
model.load_state_dict(torch.load("best_model.pth"))
>>>>>>> 1c78d8cb5cc86a3abbae0f7274ab89f5b381c2a5
model.eval()

# ---- LOAD IMAGE ----
image = Image.open(image_path).convert("RGB")
<<<<<<< HEAD
image = transform(image).unsqueeze(0).to(device)
=======
image = transform(image).unsqueeze(0)
>>>>>>> 1c78d8cb5cc86a3abbae0f7274ab89f5b381c2a5

# ---- PREDICT ----
with torch.no_grad():
    outputs = model(image)
    probs = torch.softmax(outputs, dim=1)
<<<<<<< HEAD
    confidence, pred = torch.max(probs, 1)
=======
    _, pred = torch.max(probs, 1)
>>>>>>> 1c78d8cb5cc86a3abbae0f7274ab89f5b381c2a5

classes = ["glaucoma", "normal"]

print("Prediction:", classes[pred.item()])
<<<<<<< HEAD
print("Confidence:", confidence.item())

=======
print("Confidence:", probs[0][pred.item()].item())
>>>>>>> 1c78d8cb5cc86a3abbae0f7274ab89f5b381c2a5
