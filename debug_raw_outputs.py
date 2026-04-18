#!/usr/bin/env python3
"""
Debug script to inspect raw model outputs on Im316_g_ACRIMA.jpg
Saves intermediate probability maps and masks at each stage for inspection
"""

import sys
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

# Resolve project root and add to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

from app import load_segmentation_model, load_classification_model
import torch
import torch.nn.functional as F

# Create debug output folder
DEBUG_FOLDER = PROJECT_ROOT / 'debug_raw'
DEBUG_FOLDER.mkdir(parents=True, exist_ok=True)

# Image path
candidate_paths = [
    PROJECT_ROOT / 'data' / 'test' / 'Im316_g_ACRIMA.jpg',
    PROJECT_ROOT / 'Im316_g_ACRIMA.jpg',
    PROJECT_ROOT / 'archive' / 'Database' / 'Images' / 'Im316_g_ACRIMA.jpg',
]
IMG_PATH = next((p for p in candidate_paths if p.exists()), None)
if IMG_PATH is None:
    raise FileNotFoundError(
        f"Image not found. Tried: {', '.join(str(p) for p in candidate_paths)}"
    )

print(f"[DEBUG] Loading image from {IMG_PATH}")
original_img_pil = Image.open(IMG_PATH).convert('RGB')
print(f"[DEBUG] Image size: {original_img_pil.size}")

# Load models
print("[DEBUG] Loading segmentation model...")
seg_model = load_segmentation_model()
print("[DEBUG] Segmentation model loaded")

print("[DEBUG] Loading classification model...")
clf_model = load_classification_model()
print("[DEBUG] Classification model loaded")

# Preprocess image for segmentation
input_tensor = torch.from_numpy(np.array(original_img_pil)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
print(f"[DEBUG] Input tensor shape: {input_tensor.shape}")

# Match training preprocessing: resize + ToTensor only (no normalization)
input_tensor = F.interpolate(input_tensor, size=(256, 256), mode='bilinear', align_corners=False)
print(f"[DEBUG] Preprocessed input shape: {input_tensor.shape}")

# Run model
print("[DEBUG] Running segmentation model inference...")
with torch.no_grad():
    output = seg_model(input_tensor)
    
print(f"[DEBUG] Model output type: {type(output)}")

if not isinstance(output, torch.Tensor):
    print("[DEBUG] Unexpected output type from model")
    print(f"[DEBUG] Output: {output}")
    sys.exit(1)

print(f"[DEBUG] Logits shape: {tuple(output.shape)}")
print(f"[DEBUG] Logits dtype: {output.dtype}")
print(f"[DEBUG] Logits min/max: {output.min().item():.6f} / {output.max().item():.6f}")

if output.ndim != 4 or output.shape[1] < 2:
    print(f"[DEBUG] Unexpected tensor layout. Expected [B,2,H,W], got {tuple(output.shape)}")
    sys.exit(1)

probs = torch.sigmoid(output)
disc_prob = probs[0, 0].cpu().numpy().astype(np.float32)
cup_prob = probs[0, 1].cpu().numpy().astype(np.float32)

print(f"\n[DEBUG] Disc probability map:")
print(f"  Shape: {disc_prob.shape}")
print(f"  Min: {disc_prob.min():.6f}, Max: {disc_prob.max():.6f}")
print(f"  Mean: {disc_prob.mean():.6f}, Std: {disc_prob.std():.6f}")
print(f"  Top 5 values: {np.sort(disc_prob.flatten())[-5:]}")

print(f"\n[DEBUG] Cup probability map:")
print(f"  Shape: {cup_prob.shape}")
print(f"  Min: {cup_prob.min():.6f}, Max: {cup_prob.max():.6f}")
print(f"  Mean: {cup_prob.mean():.6f}, Std: {cup_prob.std():.6f}")
print(f"  Top 5 values: {np.sort(cup_prob.flatten())[-5:]}")

# Save raw probability maps as heatmaps
print("\n[DEBUG] Saving probability heatmaps...")

def save_heatmap(prob_map, filename, title):
    fig, ax = plt.subplots(figsize=(8, 8))
    im = ax.imshow(prob_map, cmap='turbo')
    ax.set_title(title)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    filepath = DEBUG_FOLDER / filename
    plt.savefig(filepath, dpi=100)
    plt.close()
    print(f"  Saved: {filepath}")

save_heatmap(disc_prob, 'disc_raw_prob.png', 'Disc Raw Probability Map')
save_heatmap(cup_prob, 'cup_raw_prob.png', 'Cup Raw Probability Map')

# Find peaks
print("\n[DEBUG] Finding probability peaks...")
disc_peak_idx = np.unravel_index(np.argmax(disc_prob), disc_prob.shape)
cup_peak_idx = np.unravel_index(np.argmax(cup_prob), cup_prob.shape)

print(f"  Disc peak location: {disc_peak_idx} (y={disc_peak_idx[0]}, x={disc_peak_idx[1]})")
print(f"  Disc peak value: {disc_prob[disc_peak_idx]:.6f}")
print(f"  Cup peak location: {cup_peak_idx} (y={cup_peak_idx[0]}, x={cup_peak_idx[1]})")
print(f"  Cup peak value: {cup_prob[cup_peak_idx]:.6f}")

# Visualize peaks on probability map
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Disc
axes[0].imshow(disc_prob, cmap='turbo')
axes[0].plot(disc_peak_idx[1], disc_peak_idx[0], 'r+', markersize=20, markeredgewidth=3)
axes[0].set_title(f'Disc Prob + Peak\nPeak @ ({disc_peak_idx[1]}, {disc_peak_idx[0]})')
axes[0].grid(True, alpha=0.3)

# Cup
axes[1].imshow(cup_prob, cmap='turbo')
axes[1].plot(cup_peak_idx[1], cup_peak_idx[0], 'r+', markersize=20, markeredgewidth=3)
axes[1].set_title(f'Cup Prob + Peak\nPeak @ ({cup_peak_idx[1]}, {cup_peak_idx[0]})')
axes[1].grid(True, alpha=0.3)

filepath = DEBUG_FOLDER / 'probability_maps_with_peaks.png'
plt.tight_layout()
plt.savefig(filepath, dpi=100)
plt.close()
print(f"  Saved: {filepath}")

# Apply adaptive percentile thresholding
print("\n[DEBUG] Applying adaptive thresholding...")
disc_threshold = np.percentile(disc_prob, 97.2)
cup_threshold = np.percentile(cup_prob, 98.5)

print(f"  Disc threshold (97.2%ile): {disc_threshold:.6f}")
print(f"  Cup threshold (98.5%ile): {cup_threshold:.6f}")

disc_binary = (disc_prob > disc_threshold).astype(np.uint8) * 255
cup_binary = (cup_prob > cup_threshold).astype(np.uint8) * 255

print(f"  Disc binary - pixels above threshold: {np.sum(disc_binary > 0)}")
print(f"  Cup binary - pixels above threshold: {np.sum(cup_binary > 0)}")

# Save binary masks
fig, axes = plt.subplots(1, 2, figsize=(14, 6))
axes[0].imshow(disc_binary, cmap='gray')
axes[0].set_title(f'Disc Binary Mask\n({np.sum(disc_binary > 0)} pixels)')
axes[0].grid(True, alpha=0.3)

axes[1].imshow(cup_binary, cmap='gray')
axes[1].set_title(f'Cup Binary Mask\n({np.sum(cup_binary > 0)} pixels)')
axes[1].grid(True, alpha=0.3)

filepath = DEBUG_FOLDER / 'binary_masks_after_thresholding.png'
plt.tight_layout()
plt.savefig(filepath, dpi=100)
plt.close()
print(f"  Saved: {filepath}")

print("\n[DEBUG] Debug complete! Check debug_raw/ folder for outputs.")
