import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path

try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError:
    print("Installing efficientnet_pytorch...")
    os.system("pip install efficientnet_pytorch")
    from efficientnet_pytorch import EfficientNet


# ---- PATH HANDLING (Kaggle + Local) ----
if Path("/kaggle/input").exists():
    print("Running in Kaggle")
    base_path = "/kaggle/input/datasets/avinashreddy2309/glaucoma-detection/data/data"
    train_path = os.path.join(base_path, "train")
    val_path = os.path.join(base_path, "val")
    save_path = "/kaggle/working/best_model.pth"
else:
    print("Running locally")
    train_path = "data/train"
    val_path = "data/val"
    save_path = "best_model.pth"

# ---- CHECK PATHS ----
print("Train path:", train_path)
print("Val path:", val_path)
print("Train exists:", os.path.exists(train_path))
print("Val exists:", os.path.exists(val_path))

if not os.path.exists(train_path) or not os.path.exists(val_path):
    raise FileNotFoundError(
        f"Dataset paths not found. Expected train={train_path}, val={val_path}. "
        "Run split.py first to prepare the dataset."
    )

# ---- TRANSFORMS ----
train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

val_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
])

# ---- DATASET ----
train_data = datasets.ImageFolder(train_path, transform=train_transform)
val_data = datasets.ImageFolder(val_path, transform=val_transform)

print("Training images:", len(train_data))
print("Validation images:", len(val_data))
# ImageFolder sorts class names alphabetically: ['glaucoma', 'normal']
print("Classes:", train_data.classes)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader = DataLoader(val_data, batch_size=8)

# ---- MODEL ----
print("Loading model...")
model = EfficientNet.from_pretrained("efficientnet-b4")
model._fc = nn.Linear(model._fc.in_features, 2)

# ---- DEVICE ----
if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model = model.to(device)
print("Using device:", device)

# ---- LOSS + OPTIMIZER ----
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# ---- TRAINING ----
epochs = 10
best_acc = 0.0

print("Starting training...")

for epoch in range(epochs):
    print(f"\n--- Epoch {epoch + 1}/{epochs} ---")

    model.train()
    running_loss = 0.0

    for images, labels in train_loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    print(f"Training Loss: {running_loss / len(train_loader):.4f}")

    # ---- VALIDATION ----
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in val_loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = 100 * correct / total
    print(f"Validation Accuracy: {accuracy:.2f}%")

    if accuracy > best_acc:
        best_acc = accuracy
        torch.save(model.state_dict(), save_path)
        print(f"Model saved to {save_path}")

print(f"\nTraining complete. Best accuracy: {best_acc:.2f}%")
