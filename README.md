# Glaucoma Detection System

## Overview

This project detects glaucoma from retinal fundus images using two complementary deep learning workflows:

1. Image classification (`glaucoma` vs `normal`)
2. Optic disc / optic cup segmentation

The classification path gives a direct diagnostic label, while segmentation supports anatomical interpretability.

## Implemented Pipelines

### 1. Classification

- Model: EfficientNet-B0 (PyTorch)
- Training script: `train.py`
- Inference script: `predict.py`
- Classes: `glaucoma`, `normal`

### 2. Segmentation

- Model: U-Net (`UNet` in `seg_model.py`)
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

### Run segmentation on one image

```bash
python predict_seg.py
```

### Create train/val split for classification data

```bash
python split.py
```

## Notes

- `predict.py` and `predict_seg.py` open a file chooser dialog to select an input image.
- Classification and segmentation are currently trained as separate workflows.
- Segmentation masks are expected in PNG format with label convention used in `train_seg.py`.

## Future Improvements

- Add a unified pipeline that derives CDR from segmentation and feeds a final classifier.
- Add evaluation scripts (AUC, sensitivity, specificity, Dice/IoU).
- Add model checkpoint/version metadata.
