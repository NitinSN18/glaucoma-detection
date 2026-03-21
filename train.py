import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from efficientnet_pytorch import EfficientNet
import os

# ---- CHECK PATH ----
train_path = "data/train"
val_path   = "data/val"

print("Checking dataset paths...")
print("Train path exists:", os.path.exists(train_path))
print("Val path exists:", os.path.exists(val_path))

# ---- TRANSFORMS ----
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(15),
    transforms.ColorJitter(brightness=0.3, contrast=0.3),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

# ---- DATASET ----
train_data = datasets.ImageFolder(train_path, transform=transform)
val_data   = datasets.ImageFolder(val_path, transform=transform)

print("Number of training images:", len(train_data))
print("Number of validation images:", len(val_data))
print("Classes:", train_data.classes)

train_loader = DataLoader(train_data, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_data, batch_size=8)

print("Train loader batches:", len(train_loader))
print("Val loader batches:", len(val_loader))

# ---- MODEL ----
print("Loading model...")
model = EfficientNet.from_pretrained('efficientnet-b0')
model._fc = nn.Linear(model._fc.in_features, 2)
print("Model loaded")

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

print("Starting training loop...")

for epoch in range(epochs):
    print(f"\n--- Epoch {epoch+1} ---")
    
    # TRAIN
    model.train()
    running_loss = 0.0

    for i, (images, labels) in enumerate(train_loader):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()

        if (i+1) % 5 == 0:
            print(f"Batch {i+1}/{len(train_loader)}, Loss: {loss.item():.4f}")

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch {epoch+1}, Training Loss: {epoch_loss:.4f}")

    # VALIDATION
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
        torch.save(model.state_dict(), "best_model.pth")
        print("✅ Model saved!")

print("\n🎉 Training complete!")
