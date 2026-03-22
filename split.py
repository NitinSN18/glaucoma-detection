<<<<<<< HEAD
import os
import shutil
import random

# paths
source_dir = "archive/Database/Images"
train_dir = "data/train"
val_dir = "data/val"

# create folders
for split in ["train", "val"]:
    for cls in ["glaucoma", "normal"]:
        os.makedirs(f"data/{split}/{cls}", exist_ok=True)

# read files
files = os.listdir(source_dir)

# shuffle for random split
random.shuffle(files)

# split 80/20
split_idx = int(0.8 * len(files))
train_files = files[:split_idx]
val_files = files[split_idx:]

def move_files(file_list, split):
    for f in file_list:
        src = os.path.join(source_dir, f)

        if "_g_" in f:
            dst = os.path.join(f"data/{split}/glaucoma", f)
        else:
            dst = os.path.join(f"data/{split}/normal", f)

        shutil.copy(src, dst)

# move files
move_files(train_files, "train")
move_files(val_files, "val")

=======
import os
import shutil
import random

# paths
source_dir = "archive/Database/Images"
train_dir = "data/train"
val_dir = "data/val"

# create folders
for split in ["train", "val"]:
    for cls in ["glaucoma", "normal"]:
        os.makedirs(f"data/{split}/{cls}", exist_ok=True)

# read files
files = os.listdir(source_dir)

# shuffle for random split
random.shuffle(files)

# split 80/20
split_idx = int(0.8 * len(files))
train_files = files[:split_idx]
val_files = files[split_idx:]

def move_files(file_list, split):
    for f in file_list:
        src = os.path.join(source_dir, f)

        if "_g_" in f:
            dst = os.path.join(f"data/{split}/glaucoma", f)
        else:
            dst = os.path.join(f"data/{split}/normal", f)

        shutil.copy(src, dst)

# move files
move_files(train_files, "train")
move_files(val_files, "val")

>>>>>>> 1c78d8cb5cc86a3abbae0f7274ab89f5b381c2a5
print("Done splitting dataset")