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

## Usage

### Train classifier

```bash
python train.py
```

### Run classifier on one image

```bash
python predict.py
```

### Inspect EfficientNet-B4 layer and neuron counts

```bash
python model_info.py
```

This prints:
- EfficientNet-B4 base config (`include_top`)
- the repository head change (`_fc` replaced with `Linear(..., 2)`)
- layer counts using two definitions:
	- all modules except the root module
	- leaf layers (modules with no children)
- dense layer units (`out_features`, i.e., neuron count)
- total/trainable parameters and linear-layer parameter formula

Current output for this repository configuration:
- `include_top=True` in EfficientNet global params
- modules excluding root: `485`
- leaf layers: `292`
- final dense layer (`_fc`) neurons/units: `2` (`in_features=1792`, `out_features=2`)
- total/trainable parameters: `17,552,202`

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
