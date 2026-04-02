import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import cv2
from seg_model import UNet

# ---- LOAD MODEL ----
model = UNet()
model.load_state_dict(torch.load("seg_model.pth"))
model.eval()

# ---- DEVICE ----
if torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

model.to(device)

INPUT_SIZE = 256

def build_transform(size):
    return transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor()
    ])


def _fullframe_inner_ellipse_mask(shape_hw):
    h, w = shape_hw
    full_mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(0.47 * w), int(0.47 * h))
    cv2.ellipse(full_mask, center, axes, 0, 0, 360, 255, -1)
    return full_mask


def get_fundus_mask(rgb_img, roi_mode="auto"):
    """Return valid mask and selected mode: auto, blackbg, or fullframe."""
    gray = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2GRAY)
    _, rough = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

    contours, _ = cv2.findContours(rough, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    mask = np.zeros_like(rough)
    if contours:
        largest = max(contours, key=cv2.contourArea)
        cv2.drawContours(mask, [largest], -1, 255, -1)

    h, w = mask.shape
    cover_ratio = float((mask > 0).sum()) / float(h * w)

    if roi_mode == "fullframe":
        return _fullframe_inner_ellipse_mask((h, w)), "fullframe"

    # If the image is near full-frame fundus (no black background), use an inner ellipse
    # to suppress boundary activation while keeping most retina pixels.
    if roi_mode == "auto" and cover_ratio > 0.96:
        return _fullframe_inner_ellipse_mask((h, w)), "fullframe"

    # For black-background fundus, remove outer ring so the model cannot latch onto the border.
    k = max(5, int(min(h, w) * 0.025))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask, "blackbg"


def crop_to_mask(rgb_img, bin_mask, margin=0.05):
    ys, xs = np.where(bin_mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        h, w = rgb_img.shape[:2]
        return rgb_img, (0, 0, w, h)

    x1, x2 = xs.min(), xs.max()
    y1, y2 = ys.min(), ys.max()
    h, w = rgb_img.shape[:2]

    pad_x = int((x2 - x1 + 1) * margin)
    pad_y = int((y2 - y1 + 1) * margin)

    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w - 1, x2 + pad_x)
    y2 = min(h - 1, y2 + pad_y)

    return rgb_img[y1:y2 + 1, x1:x2 + 1], (x1, y1, x2 + 1, y2 + 1)


def predict_with_tta(crop_pil, model_device, input_size=INPUT_SIZE):
    """Run horizontal/vertical flip TTA and average predictions."""
    tform = build_transform(input_size)
    x0 = tform(crop_pil).unsqueeze(0).to(model_device)
    variants = [
        (x0, None),
        (torch.flip(x0, dims=[3]), "h"),
        (torch.flip(x0, dims=[2]), "v"),
    ]

    pred_sum = None
    with torch.no_grad():
        for inp, tag in variants:
            out = torch.sigmoid(model(inp))
            if tag == "h":
                out = torch.flip(out, dims=[3])
            elif tag == "v":
                out = torch.flip(out, dims=[2])

            pred_sum = out if pred_sum is None else (pred_sum + out)

    return (pred_sum / len(variants)).squeeze(0).cpu().numpy()


def predict_multiscale_tta(crop_pil, model_device, scales=(224, 256, 320)):
    """Average predictions across multiple scales plus flip-TTA."""
    pred_acc = None
    target_size = INPUT_SIZE
    for size in scales:
        p = predict_with_tta(crop_pil, model_device, input_size=size)
        if p.shape[1] != target_size or p.shape[2] != target_size:
            disc_resized = cv2.resize(p[0], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            cup_resized = cv2.resize(p[1], (target_size, target_size), interpolation=cv2.INTER_LINEAR)
            p = np.stack([disc_resized, cup_resized], axis=0)
        pred_acc = p if pred_acc is None else (pred_acc + p)
    return pred_acc / float(len(scales))


def choose_largest_component(bin_img, min_area=25):
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(bin_img)
    if not contours:
        return out

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) < min_area:
        return out

    cv2.drawContours(out, [largest], -1, 255, -1)
    return out


def choose_component_near_point(bin_img, point_xy, min_area=25):
    """Pick component closest to expected disc center to avoid distant false positives."""
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    out = np.zeros_like(bin_img)
    if not contours:
        return out

    px, py = point_xy
    best = None
    best_score = float("inf")
    for c in contours:
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        m = cv2.moments(c)
        if m["m00"] > 0:
            cx = m["m10"] / m["m00"]
            cy = m["m01"] / m["m00"]
        else:
            pts = c.reshape(-1, 2)
            cx, cy = float(pts[:, 0].mean()), float(pts[:, 1].mean())

        dist2 = (cx - px) ** 2 + (cy - py) ** 2
        area_bonus = 0.02 * area
        score = dist2 - area_bonus
        if score < best_score:
            best_score = score
            best = c

    if best is None:
        return out
    cv2.drawContours(out, [best], -1, 255, -1)
    return out


def regularize_with_ellipse(bin_img, blend=0.45):
    """Smooth jagged masks by blending with fitted ellipse when contour quality allows."""
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bin_img

    c = max(contours, key=cv2.contourArea)
    if len(c) < 5:
        return bin_img

    ellipse = cv2.fitEllipse(c)
    ellipse_mask = np.zeros_like(bin_img)
    cv2.ellipse(ellipse_mask, ellipse, 255, -1)

    merged = cv2.addWeighted(bin_img.astype(np.float32), 1.0 - blend, ellipse_mask.astype(np.float32), blend, 0)
    return (merged >= 128).astype(np.uint8) * 255


def _mask_centroid(bin_img):
    ys, xs = np.where(bin_img > 0)
    if len(xs) == 0:
        h, w = bin_img.shape
        return w // 2, h // 2
    return int(xs.mean()), int(ys.mean())


def refine_disc_from_intensity(rgb_img, valid_mask, center_xy, base_disc):
    """Expand/refine disc using local brightness contrast around ONH center."""
    h, w = valid_mask.shape
    cx, cy = center_xy
    win = int(0.45 * min(h, w))
    win = max(120, min(win, min(h, w)))

    x1 = max(0, cx - win // 2)
    y1 = max(0, cy - win // 2)
    x2 = min(w, x1 + win)
    y2 = min(h, y1 + win)

    patch = rgb_img[y1:y2, x1:x2]
    patch_valid = (valid_mask[y1:y2, x1:x2] > 0)
    if patch.size == 0 or patch_valid.sum() < 100:
        return base_disc

    l_chan = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    vals = l_chan[patch_valid]
    p2, p98 = np.percentile(vals, [2, 98])
    l_norm = np.clip((l_chan - p2) / max(1e-6, p98 - p2), 0.0, 1.0)

    thr = float(np.percentile(l_norm[patch_valid], 68))
    cand = (l_norm >= thr).astype(np.uint8) * 255
    cand = np.where(patch_valid, cand, 0).astype(np.uint8)
    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    cand = cv2.morphologyEx(cand, cv2.MORPH_CLOSE, kd)
    cand = cv2.morphologyEx(cand, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)))
    cand = choose_component_near_point(cand, (cx - x1, cy - y1), min_area=max(100, int(0.01 * patch_valid.sum())))

    full = np.zeros((h, w), dtype=np.uint8)
    full[y1:y2, x1:x2] = cand

    merged = np.where((full > 0) | (base_disc > 0), 255, 0).astype(np.uint8)
    merged = np.where(valid_mask > 0, merged, 0).astype(np.uint8)
    merged = choose_component_near_point(merged, center_xy, min_area=max(120, int(0.0007 * h * w)))
    merged = regularize_with_ellipse(merged, blend=0.60)
    return merged


def refine_cup_from_intensity(rgb_img, disc_mask, base_cup):
    """Refine cup inside disc using higher local brightness percentile."""
    if np.sum(disc_mask > 0) < 80:
        return base_cup

    l_chan = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)[:, :, 0].astype(np.float32)
    disc_vals = l_chan[disc_mask > 0]
    if disc_vals.size < 50:
        return base_cup

    p10, p99 = np.percentile(disc_vals, [10, 99])
    l_norm = np.clip((l_chan - p10) / max(1e-6, p99 - p10), 0.0, 1.0)
    cup_thr = float(np.percentile(l_norm[disc_mask > 0], 82))

    cup_cand = ((l_norm >= cup_thr) & (disc_mask > 0)).astype(np.uint8) * 255
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cup_cand = cv2.morphologyEx(cup_cand, cv2.MORPH_CLOSE, kc)
    cup_cand = cv2.morphologyEx(cup_cand, cv2.MORPH_OPEN, kc)

    dcx, dcy = _mask_centroid(disc_mask)
    cup_cand = choose_component_near_point(cup_cand, (dcx, dcy), min_area=max(30, int(0.02 * np.sum(disc_mask > 0))))
    cup_cand = regularize_with_ellipse(cup_cand, blend=0.35)
    cup_cand = np.where(disc_mask > 0, cup_cand, 0).astype(np.uint8)

    base_area = int(np.sum(base_cup > 0))
    cand_area = int(np.sum(cup_cand > 0))
    disc_area = int(np.sum(disc_mask > 0))

    # Replace if existing cup is too small/noisy and candidate has plausible area ratio.
    ratio = cand_area / max(1, disc_area)
    if base_area < max(40, int(0.05 * disc_area)) and 0.05 <= ratio <= 0.75:
        return cup_cand

    return base_cup


def adaptive_binary(prob_map, valid_mask, pct=98.0, min_thr=0.2, max_thr=0.8):
    vals = prob_map[valid_mask > 0]
    if vals.size == 0:
        return np.zeros_like(prob_map, dtype=np.uint8), min_thr

    thr = float(np.percentile(vals, pct))
    thr = max(min_thr, min(max_thr, thr))
    bin_img = (prob_map >= thr).astype(np.uint8) * 255
    return bin_img, thr


def classical_disc_cup_fallback(rgb_img, valid_mask, center_xy):
    """Fallback segmentation using local brightness around a likely disc center."""
    h, w = valid_mask.shape
    cx, cy = center_xy

    win = int(0.35 * min(h, w))
    win = max(80, min(win, min(h, w)))
    x1 = max(0, cx - win // 2)
    y1 = max(0, cy - win // 2)
    x2 = min(w, x1 + win)
    y2 = min(h, y1 + win)

    patch = rgb_img[y1:y2, x1:x2]
    patch_valid = (valid_mask[y1:y2, x1:x2] > 0)
    if patch.size == 0 or patch_valid.sum() < 50:
        return np.zeros((h, w), dtype=np.uint8), np.zeros((h, w), dtype=np.uint8)

    lab = cv2.cvtColor(patch, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0].astype(np.float32)

    vals = l_chan[patch_valid]
    p2, p98 = np.percentile(vals, [2, 98])
    l_norm = np.clip((l_chan - p2) / max(1e-6, p98 - p2), 0.0, 1.0)

    disc_thr = float(np.percentile(l_norm[patch_valid], 78))
    disc_bin = (l_norm >= disc_thr).astype(np.uint8) * 255
    disc_bin = np.where(patch_valid, disc_bin, 0).astype(np.uint8)

    kd = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    disc_bin = cv2.morphologyEx(disc_bin, cv2.MORPH_CLOSE, kd)
    disc_bin = choose_largest_component(disc_bin, min_area=max(80, int(0.01 * patch_valid.sum())))

    cup_bin = np.zeros_like(disc_bin)
    cup_vals = l_norm[disc_bin > 0]
    if cup_vals.size > 20:
        cup_thr = float(np.percentile(cup_vals, 82))
        cup_bin = ((l_norm >= cup_thr) & (disc_bin > 0)).astype(np.uint8) * 255
        kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        cup_bin = cv2.morphologyEx(cup_bin, cv2.MORPH_CLOSE, kc)
        cup_bin = choose_largest_component(cup_bin, min_area=max(30, int(0.002 * patch_valid.sum())))

    full_disc = np.zeros((h, w), dtype=np.uint8)
    full_cup = np.zeros((h, w), dtype=np.uint8)
    full_disc[y1:y2, x1:x2] = disc_bin
    full_cup[y1:y2, x1:x2] = cup_bin
    full_disc = np.where(valid_mask > 0, full_disc, 0).astype(np.uint8)
    full_cup = np.where(full_disc > 0, full_cup, 0).astype(np.uint8)

    return full_disc, full_cup


def build_disc_prior(rgb_img, valid_mask):
    """Classical prior: locate bright optic-disc-like region and build Gaussian attention map."""
    rgb = rgb_img.astype(np.float32)
    score = 0.6 * rgb[:, :, 0] + 0.4 * rgb[:, :, 1] - 0.2 * rgb[:, :, 2]
    score = np.where(valid_mask > 0, score, 0.0)

    vals = score[valid_mask > 0]
    if vals.size == 0:
        h, w = valid_mask.shape
        prior = np.ones((h, w), dtype=np.float32)
        return prior, (w // 2, h // 2), 0.0

    p2, p98 = np.percentile(vals, [2, 98])
    score = np.clip((score - p2) / (max(1e-6, p98 - p2)), 0.0, 1.0)
    sigma_blur = max(3, int(min(valid_mask.shape) * 0.015))
    if sigma_blur % 2 == 0:
        sigma_blur += 1
    score = cv2.GaussianBlur(score, (sigma_blur, sigma_blur), 0)

    masked = np.where(valid_mask > 0, score, 0.0)
    y, x = np.unravel_index(np.argmax(masked), masked.shape)
    peak = float(masked[y, x])

    h, w = valid_mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    sigma_prior = max(12.0, min(h, w) * 0.16)
    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    prior = np.exp(-dist2 / (2.0 * sigma_prior * sigma_prior)).astype(np.float32)
    prior = np.where(valid_mask > 0, prior, 0.0)

    return prior, (x, y), peak


def ask_roi_mode():
    print("\nChoose ROI mode for this image:")
    print("  1) auto      (recommended)")
    print("  2) blackbg   (circular fundus with black background)")
    print("  3) fullframe (fundus fills most of the image)")
    raw = input("Enter choice [1/2/3 or auto/blackbg/fullframe] (default: auto): ").strip().lower()

    mapping = {
        "": "auto",
        "1": "auto",
        "2": "blackbg",
        "3": "fullframe",
        "auto": "auto",
        "blackbg": "blackbg",
        "fullframe": "fullframe",
    }
    return mapping.get(raw, "auto")

# ---- LOAD IMAGE ----
import tkinter as tk
from tkinter import filedialog

root = tk.Tk()
root.withdraw()

image_path = filedialog.askopenfilename(
    title="Select a Fundus Image for Segmentation",
    filetypes=[("Image Files", "*.jpg *.jpeg *.png")]
)

if not image_path:
    print("No image selected! Exiting.")
    exit()

image = Image.open(image_path).convert("RGB")
original_width, original_height = image.size
image_np = np.array(image)

# ---- ASK ROI MODE ----
user_roi_mode = ask_roi_mode()
print(f"[DEBUG] ROI mode selected: {user_roi_mode}")

# ---- FUNDUS ROI ----
fundus_mask_full, mode_used = get_fundus_mask(image_np, roi_mode=user_roi_mode)
crop_np, (x1, y1, x2, y2) = crop_to_mask(image_np, fundus_mask_full)
crop_pil = Image.fromarray(crop_np)
fundus_cover = float((fundus_mask_full > 0).sum()) / float(original_height * original_width)
print(f"[DEBUG] Valid ROI coverage: {fundus_cover:.3f}")
print(f"[DEBUG] ROI mode used: {mode_used}")

# ---- PREDICT ON CROP WITH MULTI-SCALE TTA ----
crop_prob = predict_multiscale_tta(crop_pil, device, scales=(224, 256, 320))  # shape: (2, S, S)

# Resize probs back to crop size then to full image canvas
crop_h, crop_w = crop_np.shape[:2]
disc_crop_prob = cv2.resize(crop_prob[0], (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
cup_crop_prob = cv2.resize(crop_prob[1], (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

disc_prob = np.zeros((original_height, original_width), dtype=np.float32)
cup_prob = np.zeros((original_height, original_width), dtype=np.float32)
disc_prob[y1:y2, x1:x2] = disc_crop_prob
cup_prob[y1:y2, x1:x2] = cup_crop_prob

# Suppress background and border bias.
valid_mask = fundus_mask_full
disc_prob = disc_prob * (valid_mask / 255.0)
cup_prob = cup_prob * (valid_mask / 255.0)

# Build an image-driven optic-disc prior and blend with model outputs.
disc_prior, (prior_x, prior_y), prior_peak = build_disc_prior(image_np, valid_mask)
model_peak = float(disc_prob.max())

# Increase prior influence when model confidence is weak on unseen domains.
prior_weight = 0.35
if model_peak < 0.45:
    prior_weight = 0.60

gating_disc = np.clip(0.40 + 0.60 * disc_prior, 0.0, 1.0)
gating_cup = np.clip(0.25 + 0.75 * disc_prior, 0.0, 1.0)

disc_prob = (1.0 - prior_weight) * disc_prob + prior_weight * (disc_prob * gating_disc)
cup_prob = (1.0 - prior_weight) * cup_prob + prior_weight * (cup_prob * gating_cup)
disc_prob = np.where(valid_mask > 0, disc_prob, 0.0)
cup_prob = np.where(valid_mask > 0, cup_prob, 0.0)

# DEBUG: Print statistics about predicted probabilities
print("\n[DEBUG] Model output statistics:")
print(f"  Disc channel - Min: {disc_prob.min():.4f}, Max: {disc_prob.max():.4f}, Mean: {disc_prob.mean():.4f}")
print(f"  Cup channel - Min: {cup_prob.min():.4f}, Max: {cup_prob.max():.4f}, Mean: {cup_prob.mean():.4f}")
print(f"  Prior center - x: {prior_x}, y: {prior_y}, peak: {prior_peak:.4f}, weight: {prior_weight:.2f}")

# ---- ADAPTIVE THRESHOLDING ----
disc_peak = float(disc_prob.max())
cup_peak = float(cup_prob.max())

# Lower minimum thresholds when model confidence is weak to avoid empty masks.
disc_min_thr = 0.08 if disc_peak < 0.25 else 0.16
cup_min_thr = 0.08 if cup_peak < 0.25 else 0.16
print(f"  Dynamic thresholds floor - disc: {disc_min_thr:.2f}, cup: {cup_min_thr:.2f}")

disc_mask_raw, disc_threshold = adaptive_binary(disc_prob, valid_mask, pct=97.2, min_thr=disc_min_thr, max_thr=0.75)
cup_mask_raw, cup_threshold = adaptive_binary(cup_prob, valid_mask, pct=98.5, min_thr=cup_min_thr, max_thr=0.82)

# ---- POST-PROCESSING ----
h, w = disc_mask_raw.shape
disc_min_area = max(60, int(0.0008 * h * w))
cup_min_area = max(25, int(0.00025 * h * w))

kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

disc_mask_clean = cv2.morphologyEx(disc_mask_raw, cv2.MORPH_CLOSE, kernel_d)
disc_mask_clean = cv2.morphologyEx(disc_mask_clean, cv2.MORPH_OPEN, kernel_c)
disc_mask_clean = choose_component_near_point(disc_mask_clean, (prior_x, prior_y), min_area=disc_min_area)
disc_mask_clean = regularize_with_ellipse(disc_mask_clean, blend=0.40)
disc_mask_clean = np.where(valid_mask > 0, disc_mask_clean, 0).astype(np.uint8)

cup_mask_clean = cv2.morphologyEx(cup_mask_raw, cv2.MORPH_CLOSE, kernel_c)
cup_mask_clean = cv2.morphologyEx(cup_mask_clean, cv2.MORPH_OPEN, kernel_c)

# Cup must stay inside disc.
cup_mask_clean = np.where(disc_mask_clean > 0, cup_mask_clean, 0).astype(np.uint8)
cup_mask_clean = choose_component_near_point(cup_mask_clean, (prior_x, prior_y), min_area=cup_min_area)
cup_mask_clean = regularize_with_ellipse(cup_mask_clean, blend=0.30)
cup_mask_clean = np.where(disc_mask_clean > 0, cup_mask_clean, 0).astype(np.uint8)

disc_mask = disc_mask_clean
cup_mask = cup_mask_clean

# Shape refinement pass for better anatomical disc/cup contours.
disc_mask = refine_disc_from_intensity(image_np, valid_mask, (prior_x, prior_y), disc_mask)
cup_mask = refine_cup_from_intensity(image_np, disc_mask, cup_mask)

# If model output is tiny/empty on unseen image, use classical brightness fallback.
fallback_used = False
if np.sum(disc_mask > 0) < max(120, int(0.0004 * h * w)):
    fb_disc, fb_cup = classical_disc_cup_fallback(image_np, valid_mask, (prior_x, prior_y))
    if np.sum(fb_disc > 0) > np.sum(disc_mask > 0):
        disc_mask = fb_disc
        cup_mask = fb_cup
        fallback_used = True

# Final consistency: cup must be inside disc and should not exceed disc area.
cup_mask = np.where(disc_mask > 0, cup_mask, 0).astype(np.uint8)

# ---- CALCULATE METRICS ----
disc_area = np.sum(disc_mask > 0)
cup_area = np.sum(cup_mask > 0)

# Calculate ratios
if disc_area > 0:
    cup_disc_ratio = cup_area / disc_area
    disc_diameter_pixels = 2 * np.sqrt(disc_area / np.pi)
    cup_diameter_pixels = 2 * np.sqrt(cup_area / np.pi)
else:
    cup_disc_ratio = 0
    disc_diameter_pixels = 0
    cup_diameter_pixels = 0

# ---- PRINT RESULTS ----
print("\n" + "="*50)
print("OPTIC DISC & CUP SEGMENTATION RESULTS")
print("="*50)
print(f"Image: {image_path}")
print(f"Image Size: {original_width}x{original_height} pixels")
print(f"\nDisc Area: {disc_area} px^2")
print(f"Cup Area: {cup_area} px^2")
print(f"estimated Disc Diameter: {disc_diameter_pixels:.2f} pixels")
print(f"Estimated Cup Diameter: {cup_diameter_pixels:.2f} pixels")
print(f"\n*** CUP-TO-DISC RATIO (CDR): {cup_disc_ratio:.4f} ***")
print(f"Fallback used: {'yes' if fallback_used else 'no'}")
print("\nInterpretation:")
if cup_disc_ratio < 0.3:
    print("  -> CDR < 0.3: NORMAL (Low risk)")
elif cup_disc_ratio < 0.6:
    print("  -> CDR 0.3-0.6: MODERATE (Intermediate risk)")
else:
    print("  -> CDR >= 0.6: HIGH (Risk of glaucoma)")
print("="*50 + "\n")

# ---- VISUALIZE PROBABILITY MAPS ----
# Show what the model is actually predicting before thresholding
print("\nGenerating probability heatmaps...")
fig_prob = plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(image)
plt.imshow(disc_prob, cmap="hot", alpha=0.6, vmin=0.0, vmax=1.0)
plt.colorbar(label="Disc Probability")
plt.title("Disc Probability Heatmap")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(image)
plt.imshow(cup_prob, cmap="hot", alpha=0.6, vmin=0.0, vmax=1.0)
plt.colorbar(label="Cup Probability")
plt.title("Cup Probability Heatmap")
plt.axis("off")

plt.tight_layout()
plt.savefig("/tmp/probability_maps.png", dpi=100, bbox_inches='tight')
print("Probability heatmaps saved to /tmp/probability_maps.png")
plt.show(block=False)

# ---- VISUALIZE THRESHOLDED MASKS ----
print(f"\nApplying thresholds - Disc: >{disc_threshold}, Cup: >{cup_threshold}")
fig_thresh = plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

plt.subplot(1, 3, 2)
plt.imshow(disc_mask_raw, cmap="gray")
plt.title(f"Disc Thresholded (>{disc_threshold})")
plt.axis("off")

plt.subplot(1, 3, 3)
plt.imshow(cup_mask_raw, cmap="gray")
plt.title(f"Cup Thresholded (>{cup_threshold})")
plt.axis("off")

plt.tight_layout()
plt.show(block=False)

plt.figure(figsize=(18, 6))

# Original
plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title("Original Image")
plt.axis("off")

# Disc overlay
plt.subplot(1, 4, 2)
plt.imshow(image)
disc_colored = np.zeros_like(np.array(image))
disc_colored[disc_mask > 0] = [0, 100, 255]
plt.imshow(disc_colored, alpha=0.5)
plt.title(f"Optic Disc\n(Area: {disc_area} px²)")
plt.axis("off")

# Cup overlay
plt.subplot(1, 4, 3)
plt.imshow(image)
cup_colored = np.zeros_like(np.array(image))
cup_colored[cup_mask > 0] = [255, 100, 0]
plt.imshow(cup_colored, alpha=0.5)
plt.title(f"Optic Cup\n(Area: {cup_area} px²)")
plt.axis("off")

# Combined overlay
plt.subplot(1, 4, 4)
plt.imshow(image)
combined = np.zeros_like(np.array(image))
combined[disc_mask > 0] = [0, 100, 255]
combined[cup_mask > 0] = [255, 100, 0]
plt.imshow(combined, alpha=0.5)
plt.title(f"Combined\n(CDR: {cup_disc_ratio:.4f})")
plt.axis("off")

plt.tight_layout()
plt.show()
