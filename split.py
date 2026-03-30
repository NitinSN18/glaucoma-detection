"""split.py – Prepare the glaucoma classification dataset.

Reads images from ``archive/Database/Images`` and splits them into
``data/train``, ``data/val``, and ``data/test`` subdirectories, each with
``glaucoma/`` and ``normal/`` sub-folders for use with
``torchvision.datasets.ImageFolder``.

Label convention (based on original archive file-naming):
  - Filenames containing ``_g_`` → glaucoma class
  - All other image filenames    → normal class

Run once before training:
    python split.py
"""

import os
import shutil
import random
from pathlib import Path

# ---- PATHS ----
SOURCE_DIR = Path("archive/Database/Images")
DATA_DIR = Path("data")

# ---- SPLIT RATIOS ----
TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
# TEST_RATIO = 1 - TRAIN_RATIO - VAL_RATIO  (implicit)

RANDOM_SEED = 42

# ---- SETUP ----
if not SOURCE_DIR.exists():
    raise FileNotFoundError(
        f"Source directory not found: {SOURCE_DIR}. "
        "Ensure the 'archive' folder is present."
    )

for split in ("train", "val", "test"):
    for cls in ("glaucoma", "normal"):
        (DATA_DIR / split / cls).mkdir(parents=True, exist_ok=True)

# ---- COLLECT FILES ----
files = [
    f for f in SOURCE_DIR.iterdir()
    if f.is_file() and f.suffix.lower() in (".jpg", ".jpeg", ".png")
]

if not files:
    raise RuntimeError(f"No image files found in {SOURCE_DIR}.")

random.seed(RANDOM_SEED)
random.shuffle(files)

n = len(files)
train_end = int(TRAIN_RATIO * n)
val_end = train_end + int(VAL_RATIO * n)

splits = {
    "train": files[:train_end],
    "val": files[train_end:val_end],
    "test": files[val_end:],
}

# ---- COPY FILES ----
counts: dict = {s: {"glaucoma": 0, "normal": 0} for s in splits}

for split, file_list in splits.items():
    for f in file_list:
        # Files containing '_g_' in the name are glaucoma; the rest are normal.
        cls = "glaucoma" if "_g_" in f.name else "normal"
        dst = DATA_DIR / split / cls / f.name
        shutil.copy(f, dst)
        counts[split][cls] += 1

# ---- REPORT ----
print("Dataset split complete:")
for split, cls_counts in counts.items():
    total = sum(cls_counts.values())
    print(
        f"  {split:5s}: {total:4d} images "
        f"(glaucoma={cls_counts['glaucoma']}, normal={cls_counts['normal']})"
    )
