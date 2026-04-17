# Glaucoma Detection System

## Overview

This project detects glaucoma from retinal fundus images using two complementary deep learning workflows:

1. Image classification (`glaucoma` vs `normal`)
2. Optic disc / optic cup segmentation

The classification path gives a direct diagnostic label, while segmentation supports anatomical interpretability.

## Implemented Pipelines

### 1. Classification

- Model: EfficientNet-B4 (PyTorch)
- Training script: `train.py`
- Inference script: `predict.py`
- Classes: `glaucoma`, `normal`

### 2. Segmentation

- Models:
	- U-Net (`UNet` in `seg_model.py`)
	- DeepLabV3+ (via `segmentation-models-pytorch`)
- Training script: `train_seg.py`
- Inference/visualization script: `predict_seg.py`
- Outputs: two masks
	- Channel 0: optic disc
	- Channel 1: optic cup

## Project Structure (Current)

```
glaucoma-project/
├── train.py
├── predict.py
├── seg_model.py
├── train_seg.py
├── predict_seg.py
├── split.py
├── README.md
├── best_model.pth
├── last_model.pth
├── model.pth
├── seg_model.pth
├── archive/
│   └── Database/
│       └── Images/
├── data/
│   ├── train/
│   │   ├── glaucoma/
│   │   └── normal/
│   ├── val/
│   │   ├── glaucoma/
│   │   └── normal/
│   └── test/
└── seg_data/
		├── images/
		└── masks/
```

## Tech Stack

- Python
- PyTorch / TorchVision
- EfficientNet (`efficientnet-pytorch`)
- PIL
- NumPy
- Matplotlib
- Tkinter (file picker UI for inference scripts)

## Setup

Install dependencies with:

```bash
pip install -r requirements.txt
```

If you are using the included virtual environment, activate it first:

```bash
source .venv/bin/activate
```

Then run the scripts with `python`, not the system `python3`, so the installed packages are available.

## EfficientNet-B4 layer / neuron count (Keras reference)

The classification model is defined in `train.py`:

- `build_model()` uses `EfficientNet.from_pretrained("efficientnet-b4")`
- final classifier head is replaced with `nn.Linear(..., 2)` (2 output neurons/classes)

If you need Keras-style layer and neuron counts for reporting, run:

```bash
pip install tensorflow-cpu
python keras_efficientnetb4_stats.py
```

This prints:

- `model.summary()`
- `len(model.layers)` (top-level Keras layers)
- EfficientNetB4 submodel internal layer count (`len(base_model.layers)`)
- Dense units (neurons), e.g. `[2]`
- optional parameter counts (`total/trainable/non-trainable`)

Default output values (`--input-size 224 --num-classes 2`) are:

- Top-level Keras layers: **4**
- EfficientNetB4 internal layers: **474**
- Expanded total (EfficientNet internals + GAP + Dense): **476**
- Dense units (neurons): **[2]**

In this context:

- **Layers** = architectural building blocks (Input, EfficientNet backbone, pooling, Dense, etc.).
- **Neurons** usually means **Dense units** in the classifier head (here, final Dense has 2 units).

## Usage

### Train classifier

```bash
python train.py
```

### Run classifier on one image

```bash
python predict.py
```

### Train segmentation model

```bash
python train_seg.py
```

Train DeepLabV3+ (recommended for better cross-domain generalization):

```bash
SEG_MODEL_ARCH=deeplabv3plus SEG_MODEL_ENCODER=resnet34 python train_seg.py
```

### Run segmentation on one image

```bash
python predict_seg.py
```

Run DeepLabV3+ checkpoint on one image:

```bash
SEG_MODEL_ARCH=deeplabv3plus SEG_MODEL_ENCODER=resnet34 python predict_seg.py
```

### Create train/val split for classification data

```bash
python split.py
```

## Notes

- `predict.py` and `predict_seg.py` open a file chooser dialog to select an input image.
- Classification and segmentation are currently trained as separate workflows.
- Segmentation training now expects three folders with matching filenames:
	- `/Users/avinash/Downloads/full-fundus`
	- `/Users/avinash/Downloads/optic-cup`
	- `/Users/avinash/Downloads/optic-disc`
- Only exact filename triplets are used for training; unmatched full-fundus images are auto-skipped and reported at startup.
- Segmentation checkpoint defaults:
	- U-Net: `seg_model.pth`
	- DeepLabV3+: `seg_model_deeplabv3plus.pth`

## Future Improvements

- Add a unified pipeline that derives CDR from segmentation and feeds a final classifier.
- Add evaluation scripts (AUC, sensitivity, specificity, Dice/IoU).
- Add model checkpoint/version metadata.
