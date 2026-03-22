# Glaucoma Detection System

## Overview

This project focuses on detecting glaucoma using retinal fundus images through a combination of image processing and deep learning techniques. The system aims to assist in early diagnosis by analyzing structural changes in the optic nerve head.

A key design consideration in this project is the separation between **segmentation** and **classification**, which significantly impacts model performance and interpretability.

---

## Core Idea

### 1. Segmentation (Feature Isolation)

Segmentation is used to isolate critical regions of the eye:

* Optic Disc (OD)
* Optic Cup (OC)

Why this matters:

* Glaucoma is strongly related to the **Cup-to-Disc Ratio (CDR)**
* Raw image classification may miss these structural relationships
* Segmentation improves feature relevance and reduces noise

Insight:
A pipeline that explicitly segments OD and OC before classification is more **clinically meaningful** and often more accurate than end-to-end black-box models.

---

### 2. Classification (Decision Making)

After segmentation, the system classifies whether glaucoma is present.

Two approaches:

* Direct classification (image → label)
* Feature-based classification (CDR → label)

Better approach:

* Use segmentation output as input to classification
* Combine deep learning with domain-specific features (like CDR)

---

## Pipeline Architecture

1. Input retinal image
2. Preprocessing (resize, normalization)
3. Segmentation of optic disc and cup
4. Feature extraction (CDR calculation)
5. Classification model prediction
6. Output result (Glaucoma / Normal)

---

## Features

* Retinal image preprocessing
* Optic disc and cup segmentation
* Glaucoma classification model
* Modular pipeline design
* Extendable for clinical datasets

---

## Tech Stack

* Python
* OpenCV
* TensorFlow / Keras
* NumPy
* (Optional) Flask for interface

---

## Project Structure

```
glaucoma-detection/
│── model/              # Trained models
│── dataset/            # Input images
│── src/                # Core logic
│── templates/          # UI (if web app)
│── static/             # CSS/JS
│── main.py             # Entry point
│── requirements.txt    # Dependencies
```

---

## Installation

```
pip install -r requirements.txt
```

---

## Usage

```
python main.py
```

Provide an input retinal image and the system will output a prediction.

---

## Output

* Glaucoma / Normal classification
* (Optional) Confidence score
* (Optional) Segmented regions visualization

---

## Key Insights

* Segmentation improves interpretability and aligns with medical standards
* Pure classification models can work but lack explainability
* Hybrid models (segmentation + classification) provide the best balance
* Feature engineering (CDR) still plays a critical role alongside deep learning

---

## Limitations

* Performance depends on dataset quality
* Large models/datasets may not be included
* Requires proper preprocessing for accurate results

---

## Future Improvements

* Improve segmentation accuracy using U-Net variants
* Add real-time detection interface
* Integrate with IoT or medical devices
* Deploy as web or mobile application

---

## Team

* N$N
* Avinash Reddy Banuri
