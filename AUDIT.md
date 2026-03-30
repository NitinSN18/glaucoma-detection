# Code Audit Report — glaucoma-detection

## Summary

Two audit passes were performed on the repository. All issues found are
documented below along with their fix status.

---

## Issues Found and Fixed

### 1. All `*.py` scripts were Jupyter notebook JSON, not Python
**Files:** `train.py`, `predict.py`, `train_seg.py`, `predict_seg.py`  
**Severity:** Critical — none of the scripts could be run as `python train.py`  
**Detail:** Every `.py` file stored raw Jupyter notebook JSON (`{"metadata":...,"cells":[...]}`).
Running any of them as a Python script produced a `SyntaxError`.  
**Fix:** All four files rewritten as proper standalone Python scripts.

---

### 2. `train_seg.py` and `predict_seg.py` were byte-for-byte identical
**Files:** `train_seg.py`, `predict_seg.py`  
**Severity:** Critical — `predict_seg.py` performed training, not prediction  
**Detail:** Both files had the same MD5 hash. The prediction notebook cell was present in the
notebook source but the files were saved from the wrong cell.  
**Fix:** `predict_seg.py` now loads saved weights and visualises disc/cup masks. `train_seg.py`
trains the model and saves `seg_model.pth`.

---

### 3. `seg_model.py` was deleted, causing UNet to be copy-pasted inline twice
**Files:** `train_seg.py`, `predict_seg.py`, `seg_model.py` (absent)  
**Severity:** High — architecture duplication; risk of the two copies drifting out of sync  
**Detail:** Commit "Delete seg_model.py" removed the shared module. Both seg scripts then
contained the full `UNet` class definition inline. Any architecture change would require editing
two files.  
**Fix:** `seg_model.py` restored as the single source of truth. Both scripts now do
`from seg_model import UNet`.

---

### 4. `Image.NEAREST` removed in Pillow ≥ 10.0 (runtime crash)
**File:** `train_seg.py` line 79  
**Severity:** High — `AttributeError` crash on any modern Pillow installation  
**Detail:** `Image.NEAREST` was deprecated in Pillow 9.1 and removed in Pillow 10.0.0
(July 2023). Code using it raises:
```
AttributeError: module 'PIL.Image' has no attribute 'NEAREST'
```
**Fix:**
```python
_nearest = getattr(Image, "Resampling", Image).NEAREST
mask = mask.resize((256, 256), resample=_nearest)
```
This is backward-compatible with both old and new Pillow.

---

### 5. `torch.load` without `weights_only=True` (security + deprecation)
**Files:** `predict.py`, `predict_seg.py`  
**Severity:** High — arbitrary code execution if `.pth` file is untrusted; FutureWarning in
PyTorch ≥ 2.0; mandatory in PyTorch ≥ 2.4  
**Detail:** The default `torch.load` uses Python's `pickle`, which can execute arbitrary code
embedded in a malicious weights file.  
**Fix:** Both calls updated to:
```python
torch.load(str(model_path), map_location=device, weights_only=True)
```

---

### 6. `train.py.save` — corrupted file with the training loop duplicated three times
**File:** `train.py.save`  
**Severity:** Medium — not runnable; confusing to anyone reading the repo  
**Detail:** The file contained three stacked copies of the training loop (EfficientNet-B0 ×2,
EfficientNet-B4 ×1) concatenated without separating newlines, e.g.:
```
print("Training complete!")import torch
```
causing a `SyntaxError` on any attempt to run or import it.  
**Fix:** File deleted.

---

### 7. Spatial augmentations applied to images but not masks in `train_seg.py`
**File:** `train_seg.py`  
**Severity:** Medium — silently corrupts training by misaligning image and mask  
**Detail:** The original notebook applied `RandomHorizontalFlip` and `RandomRotation` to
images via `torchvision.transforms`, but did not apply the same transform to the
corresponding mask. This meant a flipped image was paired with an unflipped mask.  
**Fix:** Spatial augmentations removed from the image transform. The comment explains why:
> Applying them only to the image without the same transform on the mask would produce
> misaligned image-mask pairs and corrupt training.

---

### 8. `train.py` used B4 during training but comment said B0
**File:** `train.py`  
**Severity:** Medium — misleading; the `train.py.save` backup contained B0 code while the
notebook `train.py` trained with B4. `predict.py` must match exactly.  
**Fix:** Both `train.py` and `predict.py` explicitly use `efficientnet-b4` with a comment
confirming the variant must match between the two scripts.

---

### 9. No validation-only transform in `train.py` (augmentations during validation)
**File:** `train.py`  
**Severity:** Medium — random augmentations during validation make accuracy non-deterministic
across epochs (a flipped image may score differently each run)  
**Detail:** The original code used the same `transform` (with `RandomHorizontalFlip`,
`RandomRotation`, `ColorJitter`) for both training and validation datasets.  
**Fix:** `train_transform` (with augmentations) and `val_transform` (resize + normalize only)
are now separate.

---

### 10. `split.py` did not clean up before re-running (stale file accumulation)
**File:** `split.py`  
**Severity:** Low — re-running after changing ratios would leave old files from previous run  
**Fix:** Destination class directories are deleted and recreated at the start of each run.

---

### 11. `import sys` unused in `train.py`
**File:** `train.py`  
**Severity:** Low — unused import  
**Fix:** Removed.

---

## Issues Noted (not fixed — architectural decisions)

| # | Description | Reason not fixed |
|---|---|---|
| A | No test-set evaluation script | `split.py` creates `data/test` but no script runs inference over it. Adding a dedicated `evaluate.py` is outside the current scope. |
| B | Single-threaded DataLoader (`num_workers=0`) | Safe default; adding workers would require `if __name__ == "__main__":` guard which changes script structure. |
| C | No learning-rate scheduler | Training improvement, not a correctness bug. |
| D | Dice loss computed batch-wide not per-channel | Makes disc and cup losses coupled; a per-channel Dice would be more precise. Kept as-is to minimise changes. |

---

## File Status After Fixes

| File | Status |
|---|---|
| `train.py` | Rewritten — proper Python script, B4, separate train/val transforms, unused import removed |
| `predict.py` | Rewritten — `weights_only=True`, explicit CLASSES list, portable path resolution |
| `train_seg.py` | Rewritten — imports UNet from `seg_model`, Pillow-safe resize, no misaligned augmentations, MPS support |
| `predict_seg.py` | Rewritten — actually predicts (was identical to train), `weights_only=True`, imports UNet |
| `seg_model.py` | Restored — single source of truth for UNet architecture |
| `split.py` | Fixed — cleans output dirs before re-run, adds test split, uses pathlib |
| `train.py.save` | Deleted — corrupted, non-runnable duplicate |
