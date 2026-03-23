import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

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
image_path = "seg_data/images/image_0.jpg"   # change this
image = Image.open(image_path).convert("RGB")

# ---- PREPROCESS ----
x = transform(image).unsqueeze(0).to(device)

# ---- PREDICT ----
with torch.no_grad():
    mask = model(x)
    mask = torch.sigmoid(mask)
    mask = (mask > 0.5).float()

# ---- MOVE TO CPU FOR PLOTTING ----
mask = mask.squeeze().cpu().numpy()

# ---- VISUALIZE ----
plt.figure(figsize=(10,5))

# Original
plt.subplot(1,2,1)
plt.imshow(image)
plt.title("Original Image")

# Overlay
plt.subplot(1,2,2)
plt.imshow(image)
plt.imshow(mask, alpha=0.4)
plt.title("Overlay (Segmentation)")

plt.show()
