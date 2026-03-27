import os
import shutil
import random

source_dir = "archive/Database/Images"
train_dir = "data/train"
val_dir = "data/val"

# Create destination folders
for split in ["train", "val"]:
    for cls in ["glaucoma", "normal"]:
        os.makedirs(f"data/{split}/{cls}", exist_ok=True)

# Read only image files
files = [
    f for f in os.listdir(source_dir)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

# Deterministic shuffle for reproducible splits
random.seed(42)
random.shuffle(files)

split_idx = int(0.8 * len(files))
train_files = files[:split_idx]
val_files = files[split_idx:]


def move_files(file_list, split):
    for f in file_list:
        src = os.path.join(source_dir, f)
        if not os.path.isfile(src):
            continue

        if "_g_" in f:
            dst = os.path.join(f"data/{split}/glaucoma", f)
        else:
            dst = os.path.join(f"data/{split}/normal", f)

        shutil.copy(src, dst)

move_files(train_files, "train")
move_files(val_files, "val")

print(f"Done splitting dataset: train={len(train_files)}, val={len(val_files)}")