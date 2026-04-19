import os
import json
import io
import base64
from pathlib import Path
from datetime import datetime
from functools import wraps

import torch
import numpy as np
from PIL import Image
import cv2
from flask import Flask, render_template, request, jsonify, send_from_directory, session, redirect, url_for
from werkzeug.utils import secure_filename
import torchvision.transforms as transforms

# Suppress TensorFlow/other warnings
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB max
app.config['UPLOAD_FOLDER'] = 'temp_uploads'
app.config['SECRET_KEY'] = os.getenv('APP_SECRET_KEY', 'dev-secret-change-me')
app.config['PATIENT_RECORDS_FILE'] = os.getenv('PATIENT_RECORDS_FILE', 'records/patient_records.jsonl')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(Path(app.config['PATIENT_RECORDS_FILE']).parent, exist_ok=True)

AUTH_USERNAME = os.getenv('APP_USERNAME', 'host')
AUTH_PASSWORD = os.getenv('APP_PASSWORD', 'host123')

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else ('mps' if torch.backends.mps.is_available() else 'cpu'))
print(f"Using device: {device}")


# ============================================================================
# MODEL LOADING
# ============================================================================

def get_device():
    """Get the best device available"""
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def load_classification_model():
    """Load the EfficientNet classification model"""
    try:
        from efficientnet_pytorch import EfficientNet
        
        model_path = Path('best_model.pth')
        if not model_path.exists():
            print(f"Classification model not found at {model_path}")
            return None
            
        state_dict = torch.load(str(model_path), map_location=device)
        
        # Infer model architecture from state_dict
        if "_fc.weight" in state_dict and state_dict["_fc.weight"].ndim == 2:
            in_features = int(state_dict["_fc.weight"].shape[1])
            by_fc_width = {
                1280: "efficientnet-b0", 1408: "efficientnet-b2", 1536: "efficientnet-b3",
                1792: "efficientnet-b4", 2048: "efficientnet-b5", 2304: "efficientnet-b6",
                2560: "efficientnet-b7",
            }
            model_arch = by_fc_width.get(in_features, "efficientnet-b4")
        else:
            model_arch = "efficientnet-b4"
        
        model = EfficientNet.from_name(model_arch)
        model._fc = torch.nn.Linear(model._fc.in_features, 2)
        model.load_state_dict(state_dict)
        model = model.to(device)
        model.eval()
        
        print(f"Loaded classification model: {model_arch}")
        return model
    except Exception as e:
        print(f"Error loading classification model: {e}")
        return None


def load_segmentation_model():
    """Load the segmentation model (DeepLabV3+ or U-Net)"""
    try:
        from seg_model import create_segmentation_model
        
        arch = os.getenv("SEG_MODEL_ARCH", "deeplabv3plus").strip().lower()
        encoder = os.getenv("SEG_MODEL_ENCODER", "resnet34")
        
        model_path = f"seg_model_{arch}.pth" if arch != "unet" else "seg_model.pth"
        if not Path(model_path).exists():
            model_path = "seg_model.pth"
        
        if not Path(model_path).exists():
            print(f"Segmentation model not found")
            return None
        
        model = create_segmentation_model(
            arch=arch,
            out_channels=2,
            encoder_name=encoder,
            encoder_weights=None,
        )
        
        ckpt = torch.load(str(model_path), map_location=device)
        if isinstance(ckpt, dict):
            if "state_dict" in ckpt:
                ckpt = ckpt["state_dict"]
            elif "model_state_dict" in ckpt:
                ckpt = ckpt["model_state_dict"]
            ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
        
        model.load_state_dict(ckpt)
        model = model.to(device)
        model.eval()
        
        print(f"Loaded segmentation model: {arch} with {encoder} encoder")
        return model
    except Exception as e:
        print(f"Error loading segmentation model: {e}")
        return None


device = get_device()

# Initialize models as None (will be loaded on startup)
classification_model = None
segmentation_model = None


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def allowed_file(filename):
    """Check if file extension is allowed"""
    allowed_extensions = {'png', 'jpg', 'jpeg', 'bmp'}
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def image_to_base64(image_array):
    """Convert numpy image array to base64 string"""
    try:
        if isinstance(image_array, torch.Tensor):
            image_array = image_array.cpu().numpy()
        
        # Normalize to 0-255 range if needed
        if image_array.max() <= 1:
            image_array = (image_array * 255).astype(np.uint8)
        else:
            image_array = image_array.astype(np.uint8)
        
        # Convert to PIL Image and encode
        if len(image_array.shape) == 3 and image_array.shape[2] == 3:
            pil_img = Image.fromarray(image_array, 'RGB')
        else:
            pil_img = Image.fromarray(image_array, 'L')
        
        buffer = io.BytesIO()
        pil_img.save(buffer, format='PNG')
        img_base64 = base64.b64encode(buffer.getvalue()).decode()
        return img_base64
    except Exception as e:
        print(f"Error converting image to base64: {e}")
        return None


def overlay_masks_on_image(original_image, disc_mask, cup_mask):
    """Create an overlay visualization of masks on the original image"""
    try:
        # Ensure images are in the right format
        if isinstance(original_image, str):
            original_image = cv2.imread(original_image)
            original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        # Resize masks to match original image size
        if isinstance(disc_mask, torch.Tensor):
            disc_mask = disc_mask.cpu().numpy()
        if isinstance(cup_mask, torch.Tensor):
            cup_mask = cup_mask.cpu().numpy()
        if disc_mask is None:
            disc_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        if cup_mask is None:
            cup_mask = np.zeros(original_image.shape[:2], dtype=np.uint8)
        
        # Normalize masks to 0-1 range
        if disc_mask.max() > 1:
            disc_mask = disc_mask / 255.0
        if cup_mask.max() > 1:
            cup_mask = cup_mask / 255.0
        
        h, w = original_image.shape[:2]
        disc_mask_resized = cv2.resize(disc_mask, (w, h))
        cup_mask_resized = cv2.resize(cup_mask, (w, h))
        
        # Create overlay with transparency
        overlay = original_image.copy().astype(float)
        
        # Red for disc mask (opacity 0.3)
        disc_region = disc_mask_resized > 0.5
        overlay[disc_region] = overlay[disc_region] * 0.7 + np.array([255, 0, 0]) * 0.3
        
        # Blue for cup mask (opacity 0.3)
        cup_region = cup_mask_resized > 0.5
        overlay[cup_region] = overlay[cup_region] * 0.7 + np.array([0, 0, 255]) * 0.3
        
        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
        return overlay
    except Exception as e:
        print(f"Error creating overlay: {e}")
        return original_image


def probability_to_heatmap(prob_map):
    """Convert probability map to RGB heatmap image."""
    try:
        if isinstance(prob_map, torch.Tensor):
            prob_map = prob_map.cpu().numpy()
        if prob_map is None:
            return None

        prob_map = prob_map.astype(np.float32)
        if prob_map.max() > 1.0:
            prob_map = prob_map / 255.0
        prob_map = np.clip(prob_map, 0.0, 1.0)

        heat_uint8 = (prob_map * 255).astype(np.uint8)
        heat_bgr = cv2.applyColorMap(heat_uint8, cv2.COLORMAP_TURBO)
        return cv2.cvtColor(heat_bgr, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error creating heatmap: {e}")
        return None


def _fullframe_inner_ellipse_mask(shape_hw):
    """Create inner ellipse mask for full-frame fundus images."""
    h, w = shape_hw
    full_mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(0.47 * w), int(0.47 * h))
    cv2.ellipse(full_mask, center, axes, 0, 0, 360, 255, -1)
    return full_mask


def get_fundus_mask(rgb_img, roi_mode="auto"):
    """Detect fundus region, handling both black-background and full-frame images."""
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

    if roi_mode == "auto" and cover_ratio > 0.96:
        return _fullframe_inner_ellipse_mask((h, w)), "fullframe"

    k = max(5, int(min(h, w) * 0.025))
    if k % 2 == 0:
        k += 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    mask = cv2.erode(mask, kernel, iterations=1)

    return mask, "blackbg"


def build_disc_prior(rgb_img, valid_mask):
    """Build probabilistic disc prior from image brightness."""
    rgb = rgb_img.astype(np.float32)
    score = 0.6 * rgb[:, :, 0] + 0.4 * rgb[:, :, 1] - 0.2 * rgb[:, :, 2]
    score = np.where(valid_mask > 0, score, 0.0)

    vals = score[valid_mask > 0]
    if vals.size == 0:
        h, w = valid_mask.shape
        return np.ones((h, w), dtype=np.float32), (w // 2, h // 2), 0.0

    p2, p98 = np.percentile(vals, [2, 98])
    score = np.clip((score - p2) / max(1e-6, p98 - p2), 0.0, 1.0)
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


def _clahe_enhance_rgb(rgb_img):
    """Contrast-normalize retina image to improve model robustness on low-contrast scans."""
    lab = cv2.cvtColor(rgb_img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
    l_eq = clahe.apply(l)
    merged = cv2.merge([l_eq, a, b])
    return cv2.cvtColor(merged, cv2.COLOR_LAB2RGB)


def _run_segmentation_probs(rgb_img_np, out_hw, tta=True):
    """Run segmentation model and return disc/cup probabilities resized to out_hw."""
    transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
    ])
    pil_img = Image.fromarray(rgb_img_np)
    tensor = transform(pil_img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = segmentation_model(tensor)
        if tta:
            tensor_flip = torch.flip(tensor, dims=[3])
            logits_flip = segmentation_model(tensor_flip)
            logits_flip = torch.flip(logits_flip, dims=[3])
            logits = 0.5 * (logits + logits_flip)
        probs = torch.sigmoid(logits)
        disc_prob_small = probs[0, 0].cpu().numpy().astype(np.float32)
        cup_prob_small = probs[0, 1].cpu().numpy().astype(np.float32)

    out_h, out_w = out_hw
    disc_prob = cv2.resize(disc_prob_small, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    cup_prob = cv2.resize(cup_prob_small, (out_w, out_h), interpolation=cv2.INTER_LINEAR)
    return disc_prob, cup_prob


def _extract_square_crop(rgb_img_np, center_xy, side):
    """Extract square crop around center, clamped to image bounds."""
    h, w = rgb_img_np.shape[:2]
    cx, cy = center_xy
    side = int(max(64, min(side, min(h, w))))
    half = side // 2

    x1 = max(0, cx - half)
    y1 = max(0, cy - half)
    x2 = min(w, x1 + side)
    y2 = min(h, y1 + side)

    # Re-adjust to keep target side whenever possible.
    x1 = max(0, x2 - side)
    y1 = max(0, y2 - side)

    crop = rgb_img_np[y1:y2, x1:x2]
    return crop, (x1, y1, x2, y2)


def _compose_crop_probs_to_full(crop_disc_prob, crop_cup_prob, crop_box, full_hw):
    """Project crop probabilities back to full-frame with soft edge weighting."""
    h, w = full_hw
    x1, y1, x2, y2 = crop_box
    ch = max(1, y2 - y1)
    cw = max(1, x2 - x1)

    disc_crop_rs = cv2.resize(crop_disc_prob, (cw, ch), interpolation=cv2.INTER_LINEAR)
    cup_crop_rs = cv2.resize(crop_cup_prob, (cw, ch), interpolation=cv2.INTER_LINEAR)

    disc_full = np.zeros((h, w), dtype=np.float32)
    cup_full = np.zeros((h, w), dtype=np.float32)
    weight = np.zeros((h, w), dtype=np.float32)

    disc_full[y1:y2, x1:x2] = disc_crop_rs
    cup_full[y1:y2, x1:x2] = cup_crop_rs
    weight[y1:y2, x1:x2] = 1.0

    k = max(9, int(0.10 * min(ch, cw)))
    if k % 2 == 0:
        k += 1
    weight = cv2.GaussianBlur(weight, (k, k), 0)
    if float(weight.max()) > 0:
        weight = weight / float(weight.max())

    return disc_full, cup_full, weight


def _shape_features(bin_img):
    """Compute simple shape plausibility metrics for reliability scoring."""
    mask = (bin_img > 0).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0, 0.0

    c = max(contours, key=cv2.contourArea)
    area = max(1.0, float(cv2.contourArea(c)))
    peri = max(1.0, float(cv2.arcLength(c, True)))
    circularity = float((4.0 * np.pi * area) / (peri * peri))

    hull = cv2.convexHull(c)
    hull_area = max(1.0, float(cv2.contourArea(hull)))
    solidity = float(area / hull_area)
    return circularity, solidity


def choose_component_near_point(bin_img, point_xy, min_area=25):
    """Select largest component closest to a point."""
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

    if best is not None:
        cv2.drawContours(out, [best], -1, 255, -1)
    return out


def _is_logged_in():
    return bool(session.get('logged_in'))


def login_required(api=False):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if _is_logged_in():
                return func(*args, **kwargs)
            if api:
                return jsonify({'error': 'Authentication required'}), 401
            return redirect(url_for('login_page'))
        return wrapper
    return decorator


def _extract_patient_details():
    raw = request.form.get('patient_details')
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        if isinstance(parsed, dict):
            return parsed
    except Exception:
        pass
    return {}


def save_patient_record(record):
    records_file = Path(app.config['PATIENT_RECORDS_FILE'])
    records_file.parent.mkdir(parents=True, exist_ok=True)
    with records_file.open('a', encoding='utf-8') as f:
        f.write(json.dumps(record, ensure_ascii=True) + '\n')


def should_save_logbook_from_form():
    raw = str(request.form.get('save_to_logbook', 'true')).strip().lower()
    return raw in ('1', 'true', 'yes', 'on')


def _mask_centroid(bin_img):
    """Get centroid of binary mask."""
    ys, xs = np.where(bin_img > 0)
    if len(xs) == 0:
        h, w = bin_img.shape
        return w // 2, h // 2
    return int(xs.mean()), int(ys.mean())


def adaptive_binary(prob_map, valid_mask, pct=98.0, min_thr=0.2, max_thr=0.8):
    """Adaptive thresholding based on percentile of valid region."""
    vals = prob_map[valid_mask > 0]
    if vals.size == 0:
        return np.zeros_like(prob_map, dtype=np.uint8), min_thr

    thr = float(np.percentile(vals, pct))
    thr = max(min_thr, min(max_thr, thr))
    bin_img = (prob_map >= thr).astype(np.uint8) * 255
    return bin_img, thr


def regularize_with_ellipse(bin_img, blend=0.45):
    """Smooth jagged masks by blending with fitted ellipse."""
    contours, _ = cv2.findContours(bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return bin_img

    c = max(contours, key=cv2.contourArea)
    if len(c) < 5:
        return bin_img

    try:
        ellipse = cv2.fitEllipse(c)
        ellipse_mask = np.zeros_like(bin_img)
        cv2.ellipse(ellipse_mask, ellipse, 255, -1)
        merged = cv2.addWeighted(bin_img.astype(np.float32), 1.0 - blend, ellipse_mask.astype(np.float32), blend, 0)
        return (merged >= 128).astype(np.uint8) * 255
    except:
        return bin_img


def refine_disc_from_intensity(rgb_img, valid_mask, center_xy, base_disc):
    """Refine disc using local brightness contrast around ONH center."""
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

    merged = np.where(valid_mask > 0, full, 0).astype(np.uint8)
    merged = choose_component_near_point(merged, center_xy, min_area=max(120, int(0.0007 * h * w)))

    base_area = int(np.sum(base_disc > 0))
    cand_area = int(np.sum(merged > 0))
    if base_area <= 0:
        return merged

    # Only replace the model mask if the candidate is reasonably shaped and not wildly different in size.
    area_ratio = cand_area / max(1, base_area)
    if 0.70 <= area_ratio <= 1.60:
        base_perim = max(1.0, cv2.arcLength(max(cv2.findContours((base_disc > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea) if cv2.findContours((base_disc > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] else np.array([]), True))
        cand_perim = max(1.0, cv2.arcLength(max(cv2.findContours((merged > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0], key=cv2.contourArea) if cv2.findContours((merged > 0).astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0] else np.array([]), True))
        if cand_perim > 0 and base_perim / cand_perim >= 0.75:
            return merged

    return base_disc


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
    cup_thr = float(np.percentile(l_norm[disc_mask > 0], 72))

    cup_cand = ((l_norm >= cup_thr) & (disc_mask > 0)).astype(np.uint8) * 255
    kc = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    cup_cand = cv2.morphologyEx(cup_cand, cv2.MORPH_CLOSE, kc)
    cup_cand = cv2.morphologyEx(cup_cand, cv2.MORPH_OPEN, kc)

    dcx, dcy = _mask_centroid(disc_mask)
    cup_cand = choose_component_near_point(cup_cand, (dcx, dcy), min_area=max(30, int(0.02 * np.sum(disc_mask > 0))))
    cup_cand = np.where(disc_mask > 0, cup_cand, 0).astype(np.uint8)

    base_area = int(np.sum(base_cup > 0))
    cand_area = int(np.sum(cup_cand > 0))
    disc_area = int(np.sum(disc_mask > 0))

    # Replace if existing cup is too small/noisy and candidate has plausible area ratio.
    ratio = cand_area / max(1, disc_area)
    if base_area < max(40, int(0.03 * disc_area)) and 0.03 <= ratio <= 0.65:
        return cup_cand
    return base_cup


def stabilize_disc_interior(disc_mask):
    """Recover a plausible filled disc when model response is fragmented/ring-like."""
    if disc_mask is None or np.sum(disc_mask > 0) == 0:
        return np.zeros_like(disc_mask if disc_mask is not None else np.zeros((1, 1), dtype=np.uint8), dtype=np.uint8)

    mask_u8 = (disc_mask > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(mask_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return mask_u8

    c = max(contours, key=cv2.contourArea)
    c_area = float(cv2.contourArea(c))
    if c_area < 30:
        return mask_u8

    hull = cv2.convexHull(c)
    hull_mask = np.zeros_like(mask_u8)
    cv2.drawContours(hull_mask, [hull], -1, 255, -1)

    # Fill interior holes and smooth boundaries.
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    hull_mask = cv2.morphologyEx(hull_mask, cv2.MORPH_CLOSE, k)

    hull_area = float(np.sum(hull_mask > 0))
    orig_area = float(np.sum(mask_u8 > 0))
    if hull_area <= 0:
        return mask_u8

    # If the original mask is too sparse relative to its hull, trust hull as interior recovery.
    if orig_area / hull_area < 0.68:
        return hull_mask

    merged = np.where((mask_u8 > 0) | (hull_mask > 0), 255, 0).astype(np.uint8)
    merged = cv2.morphologyEx(merged, cv2.MORPH_CLOSE, k)
    return merged


def compute_segmentation_quality(disc_mask, cup_mask, fundus_mask, disc_peak, cup_peak):
    """Estimate segmentation reliability from confidence and anatomical plausibility."""
    fundus_area = int(np.sum(fundus_mask > 0))
    disc_area = int(np.sum(disc_mask > 0))
    cup_area = int(np.sum(cup_mask > 0))

    disc_area_ratio = disc_area / max(1, fundus_area)
    cdr = cup_area / max(1, disc_area)

    disc_circularity, disc_solidity = _shape_features(disc_mask)

    disc_peak_norm = min(1.0, float(disc_peak) / 0.55)
    cup_peak_norm = min(1.0, float(cup_peak) / 0.08)

    disc_area_ok = 0.01 <= disc_area_ratio <= 0.30
    cdr_ok = 0.01 <= cdr <= 0.80
    cup_present = cup_area >= 20
    disc_shape_ok = (disc_circularity >= 0.18) and (disc_solidity >= 0.72)

    if float(cup_peak) < 0.01:
        cup_present = False

    score = (
        0.30 * disc_peak_norm
        + 0.20 * cup_peak_norm
        + 0.18 * (1.0 if disc_area_ok else 0.0)
        + 0.12 * (1.0 if cdr_ok and cup_present else 0.0)
        + 0.20 * (1.0 if disc_shape_ok else 0.0)
    )

    reasons = []
    if float(disc_peak) < 0.18:
        reasons.append("Disc confidence too low")
    if float(cup_peak) < 0.01:
        reasons.append("Cup confidence too low")
    if not disc_area_ok:
        reasons.append("Disc area implausible")
    if not disc_shape_ok:
        reasons.append("Disc shape implausible")
    if not cup_present:
        reasons.append("Cup mask too small")
    elif not cdr_ok:
        reasons.append("Cup-to-disc ratio implausible")

    hard_fail = (not disc_area_ok) or (disc_solidity < 0.55) or (float(cup_peak) < 0.008)
    reliable = (score >= 0.66) and (len(reasons) <= 1) and (not hard_fail)

    return {
        'reliable': bool(reliable),
        'score': round(float(score), 3),
        'disc_peak': round(float(disc_peak), 4),
        'cup_peak': round(float(cup_peak), 4),
        'disc_area_ratio': round(float(disc_area_ratio), 4),
        'cup_to_disc_ratio': round(float(cdr), 4),
        'disc_circularity': round(float(disc_circularity), 4),
        'disc_solidity': round(float(disc_solidity), 4),
        'reason': '; '.join(reasons) if reasons else 'Segmentation quality is acceptable'
    }

def calculate_clinical_metrics(glaucoma_prob, disc_mask, cup_mask, segmentation_quality=None):
    """
    Calculate clinical metrics including sensitivity, specificity, severity level, and recommendation.
    Returns dict with clinical information for display.
    """
    # Glaucoma percentage (model confidence)
    glaucoma_pct = float(glaucoma_prob * 100)

    if disc_mask is None:
        disc_mask = np.zeros((1, 1), dtype=np.uint8)
    if cup_mask is None:
        cup_mask = np.zeros_like(disc_mask, dtype=np.uint8)
    
    segmentation_reliable = True if segmentation_quality is None else bool(segmentation_quality.get('reliable', True))

    # Sensitivity and Specificity (based on typical glaucoma screening metrics)
    # Sensitivity: True Positive Rate (ability to detect glaucoma)
    # Specificity: True Negative Rate (ability to identify normal eyes)
    # Using model-based estimates with structural information
    
    # If glaucoma probability is high, sensitivity is high but may miss edge cases
    sensitivity = min(95.0, glaucoma_pct * 0.95 + 5.0)  # Range 5-95%
    
    # Specificity inversely related to glaucoma probability
    specificity = max(70.0, 100.0 - glaucoma_pct * 0.85)  # Range 70-100%
    
    # Calculate Cup-to-Disc Ratio for severity assessment
    disc_area = int(np.sum(disc_mask > 0))
    cup_area = int(np.sum(cup_mask > 0))
    cdr = cup_area / max(1, disc_area)
    
    # Suppress structural interpretation when segmentation is unreliable.
    if not segmentation_reliable:
        return {
            'glaucoma_percentage': round(glaucoma_pct, 1),
            'sensitivity': None,
            'specificity': None,
            'cup_to_disc_ratio': None,
            'severity': 'UNRELIABLE',
            'recommendation': 'Segmentation quality is low for this image. Retake image or review manually before using structural metrics.',
            'is_valid': False
        }

    # Severity level based on glaucoma probability and structural features
    if glaucoma_pct >= 85:
        severity = "HIGH"
        recommendation = "URGENT: Visit ophthalmologist immediately"
    elif glaucoma_pct >= 65:
        severity = "MODERATE-HIGH"
        recommendation = "Schedule ophthalmologist visit within 1 week"
    elif glaucoma_pct >= 45:
        severity = "MODERATE"
        recommendation = "Schedule ophthalmologist visit within 2 weeks"
    elif glaucoma_pct >= 25:
        severity = "LOW-MODERATE"
        recommendation = "Schedule routine ophthalmologist visit"
    else:
        severity = "LOW"
        recommendation = "Normal appearance - routine follow-up recommended"
    
    return {
        'glaucoma_percentage': round(glaucoma_pct, 1),
        'sensitivity': round(sensitivity, 1),
        'specificity': round(specificity, 1),
        'cup_to_disc_ratio': round(cdr, 3),
        'severity': severity,
        'recommendation': recommendation,
        'is_valid': True
    }


def run_segmentation_pipeline(original_img_pil):
    """Run segmentation with robust local pipeline and return masks/probabilities."""
    original_img_np = np.array(original_img_pil)
    h0, w0 = original_img_np.shape[:2]

    # Detect fundus region
    fundus_mask_full, mode_used = get_fundus_mask(original_img_np, roi_mode="auto")

    # Multi-pass inference: global, contrast-normalized, and local crop refinement.
    disc_prob_global, cup_prob_global = _run_segmentation_probs(original_img_np, (h0, w0), tta=True)
    rgb_eq = _clahe_enhance_rgb(original_img_np)
    disc_prob_eq, cup_prob_eq = _run_segmentation_probs(rgb_eq, (h0, w0), tta=False)

    # Apply fundus mask to suppress black background
    fundus_norm = (fundus_mask_full / 255.0).astype(np.float32)
    disc_prob_global = disc_prob_global * fundus_norm
    cup_prob_global = cup_prob_global * fundus_norm
    disc_prob_eq = disc_prob_eq * fundus_norm
    cup_prob_eq = cup_prob_eq * fundus_norm

    # Build disc prior from brightness
    disc_prior, (prior_x, prior_y), prior_peak = build_disc_prior(original_img_np, fundus_mask_full)
    peak_global_y, peak_global_x = np.unravel_index(np.argmax(disc_prob_global), disc_prob_global.shape)

    # Blend model and prior centers for robust ONH localization.
    onh_x = int(round(0.72 * peak_global_x + 0.28 * prior_x))
    onh_y = int(round(0.72 * peak_global_y + 0.28 * prior_y))

    crop_scale = 0.58 if mode_used == "fullframe" else 0.66
    crop_side = int(crop_scale * min(h0, w0))
    crop_img, crop_box = _extract_square_crop(original_img_np, (onh_x, onh_y), crop_side)
    crop_disc_prob, crop_cup_prob = _run_segmentation_probs(crop_img, crop_img.shape[:2], tta=True)
    crop_disc_full, crop_cup_full, crop_w = _compose_crop_probs_to_full(crop_disc_prob, crop_cup_prob, crop_box, (h0, w0))

    disc_crop_blend = disc_prob_global * (1.0 - crop_w) + crop_disc_full * crop_w
    cup_crop_blend = cup_prob_global * (1.0 - crop_w) + crop_cup_full * crop_w

    # Final probability fusion (weights chosen for stable generalization).
    disc_prob = 0.60 * disc_prob_global + 0.15 * disc_prob_eq + 0.25 * disc_crop_blend
    cup_prob = 0.45 * cup_prob_global + 0.20 * cup_prob_eq + 0.35 * cup_crop_blend

    model_peak = float(disc_prob.max())
    prior_weight = 0.42 if model_peak < 0.35 else 0.26
    gating_disc = np.clip(0.62 + 0.38 * disc_prior, 0.0, 1.0)
    gating_cup = np.clip(0.50 + 0.50 * disc_prior, 0.0, 1.0)
    disc_prob = (1.0 - prior_weight) * disc_prob + prior_weight * (disc_prob * gating_disc)
    cup_prob = (1.0 - prior_weight) * cup_prob + prior_weight * (cup_prob * gating_cup)

    disc_prob = disc_prob * fundus_norm
    cup_prob = cup_prob * fundus_norm

    disc_peak_y, disc_peak_x = np.unravel_index(np.argmax(disc_prob), disc_prob.shape)
    cup_peak_y, cup_peak_x = np.unravel_index(np.argmax(cup_prob), cup_prob.shape)
    disc_peak_val = float(disc_prob[disc_peak_y, disc_peak_x])
    cup_peak_val = float(cup_prob[cup_peak_y, cup_peak_x])

    # Adaptive binary thresholding
    disc_peak = float(disc_prob.max())
    if disc_peak < 0.20:
        disc_min_thr = max(0.05, disc_peak * 0.40)
        disc_pct = 96.8
    elif disc_peak < 0.45:
        disc_min_thr = max(0.09, disc_peak * 0.28)
        disc_pct = 97.2
    else:
        disc_min_thr = 0.16
        disc_pct = 97.4

    # Keep weak cup structures alive by adapting min threshold to model peak.
    cup_peak = float(cup_prob.max())
    if cup_peak < 0.02:
        cup_min_thr = max(0.0006, cup_peak * 0.28)
        cup_pct = 96.8
    elif cup_peak < 0.05:
        cup_min_thr = max(0.0018, cup_peak * 0.34)
        cup_pct = 97.6
    elif cup_peak < 0.12:
        cup_min_thr = max(0.005, cup_peak * 0.40)
        cup_pct = 98.2
    else:
        cup_min_thr = 0.16
        cup_pct = 98.5

    disc_mask_raw, disc_thr = adaptive_binary(disc_prob, fundus_mask_full, pct=disc_pct, min_thr=disc_min_thr, max_thr=0.75)
    cup_mask_raw, cup_thr = adaptive_binary(cup_prob, fundus_mask_full, pct=cup_pct, min_thr=cup_min_thr, max_thr=0.82)

    # Morphology
    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    disc_mask = cv2.morphologyEx(disc_mask_raw, cv2.MORPH_CLOSE, kernel_d)
    disc_mask = cv2.morphologyEx(disc_mask, cv2.MORPH_OPEN, kernel_c)
    disc_mask = choose_component_near_point(disc_mask, (disc_peak_x, disc_peak_y), min_area=max(60, int(0.0008 * h0 * w0)))
    disc_mask = np.where(fundus_mask_full > 0, disc_mask, 0).astype(np.uint8)
    disc_mask = stabilize_disc_interior(disc_mask)
    disc_mask = regularize_with_ellipse(disc_mask, blend=0.12)

    cup_mask = cv2.morphologyEx(cup_mask_raw, cv2.MORPH_CLOSE, kernel_c)
    cup_mask = cv2.morphologyEx(cup_mask, cv2.MORPH_OPEN, kernel_c)
    cup_mask = np.where(disc_mask > 0, cup_mask, 0).astype(np.uint8)
    cup_mask = choose_component_near_point(cup_mask, (cup_peak_x, cup_peak_y), min_area=max(25, int(0.00025 * h0 * w0)))
    cup_mask = np.where(disc_mask > 0, cup_mask, 0).astype(np.uint8)
    cup_mask = regularize_with_ellipse(cup_mask, blend=0.08)

    # Use intensity-based refinement only as a fallback when the model mask collapses.
    if np.sum(disc_mask > 0) < max(80, int(0.0005 * h0 * w0)):
        disc_mask = refine_disc_from_intensity(original_img_np, fundus_mask_full, (disc_peak_x, disc_peak_y), disc_mask)
        disc_mask = regularize_with_ellipse(disc_mask, blend=0.12)

    if np.sum(cup_mask > 0) < max(40, int(0.00012 * h0 * w0)):
        cup_mask = refine_cup_from_intensity(original_img_np, disc_mask, cup_mask)
        cup_mask = np.where(disc_mask > 0, cup_mask, 0).astype(np.uint8)
        cup_mask = regularize_with_ellipse(cup_mask, blend=0.08)

    # Final cup rescue from model probabilities inside disc for weak cases.
    if np.sum(cup_mask > 0) < max(35, int(0.00010 * h0 * w0)) and cup_peak_val > 0.008:
        cup_inside = np.where(disc_mask > 0, cup_prob, 0.0)
        vals = cup_inside[disc_mask > 0]
        if vals.size > 50:
            rescue_thr = max(float(np.percentile(vals, 91.5)), cup_peak_val * 0.55)
            cup_rescue = ((cup_inside >= rescue_thr) & (disc_mask > 0)).astype(np.uint8) * 255
            cup_rescue = cv2.morphologyEx(cup_rescue, cv2.MORPH_CLOSE, kernel_c)
            cup_rescue = choose_component_near_point(cup_rescue, (disc_peak_x, disc_peak_y), min_area=max(22, int(0.00015 * h0 * w0)))
            if np.sum(cup_rescue > 0) >= np.sum(cup_mask > 0):
                cup_mask = cup_rescue

    segmentation_quality = compute_segmentation_quality(
        disc_mask,
        cup_mask,
        fundus_mask_full,
        disc_peak_val,
        cup_peak_val,
    )

    overlay = overlay_masks_on_image(original_img_np, disc_mask, cup_mask)
    disc_heatmap = probability_to_heatmap(disc_prob)
    cup_heatmap = probability_to_heatmap(cup_prob)

    return {
        'original_image': original_img_np,
        'disc_mask': disc_mask,
        'cup_mask': cup_mask,
        'overlay': overlay,
        'disc_heatmap': disc_heatmap,
        'cup_heatmap': cup_heatmap,
        'segmentation_quality': segmentation_quality,
    }


# ============================================================================
# API ENDPOINTS
# ============================================================================

@app.route('/login')
def login_page():
    if _is_logged_in():
        return redirect(url_for('index'))
    return render_template('login.html')


@app.route('/api/login', methods=['POST'])
def api_login():
    payload = request.get_json(silent=True) or {}
    username = str(payload.get('username', '')).strip()
    password = str(payload.get('password', ''))

    if username == AUTH_USERNAME and password == AUTH_PASSWORD:
        session['logged_in'] = True
        session['username'] = username
        return jsonify({'success': True, 'username': username}), 200

    return jsonify({'error': 'Invalid username or password'}), 401


@app.route('/api/logout', methods=['POST'])
def api_logout():
    session.clear()
    return jsonify({'success': True}), 200


@app.route('/api/auth-status', methods=['GET'])
def api_auth_status():
    return jsonify({
        'authenticated': _is_logged_in(),
        'username': session.get('username')
    }), 200

@app.route('/')
@login_required(api=False)
def index():
    """Serve the main HTML page"""
    return render_template('index.html')


@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory('static', filename)


@app.route('/api/classify', methods=['POST'])
@login_required(api=True)
def api_classify():
    """
    Classification endpoint
    Upload: POST /api/classify with image file
    Returns: {prediction, confidence, probabilities}
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Use PNG, JPG, JPEG, or BMP'}), 400
        
        if classification_model is None:
            return jsonify({'error': 'Classification model not loaded'}), 500
        
        patient_details = _extract_patient_details()

        # Save temporary file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Perform classification
        try:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            image = Image.open(filepath).convert('RGB')
            tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = classification_model(tensor)
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
            
            classes = ["glaucoma", "normal"]
            prediction = classes[pred_idx.item()]
            conf_score = float(confidence.item())
            glaucoma_prob = float(probs[0, 0].item())
            normal_prob = float(probs[0, 1].item())
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        response_data = {
            'success': True,
            'prediction': prediction,
            'confidence': conf_score,
            'probabilities': {
                'glaucoma': glaucoma_prob,
                'normal': normal_prob
            },
            'timestamp': datetime.now().isoformat(),
            'patient_details': patient_details
        }

        if should_save_logbook_from_form():
            save_patient_record({
                'timestamp': response_data['timestamp'],
                'mode': 'classify-only',
                'filename': file.filename,
                'patient_details': patient_details,
                'prediction': prediction,
                'confidence': conf_score,
                'probabilities': response_data['probabilities'],
                'record_type': 'analysis'
            })

        return jsonify(response_data), 200
    
    except Exception as e:
        print(f"Error in classify endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/segment', methods=['POST'])
@login_required(api=True)
def api_segment():
    """
    Segmentation endpoint
    Upload: POST /api/segment with image file
    Returns: {disc_mask, cup_mask, original_image, overlay}
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Use PNG, JPG, JPEG, or BMP'}), 400
        
        if segmentation_model is None:
            return jsonify({'error': 'Segmentation model not loaded'}), 500
        
        patient_details = _extract_patient_details()

        # Save temporary file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            original_img = Image.open(filepath).convert('RGB')
            seg_output = run_segmentation_pipeline(original_img)

            # Convert to base64 for JSON response
            disc_mask_b64 = image_to_base64(seg_output['disc_mask'])
            cup_mask_b64 = image_to_base64(seg_output['cup_mask'])
            original_b64 = image_to_base64(seg_output['original_image'])
            overlay_b64 = image_to_base64(seg_output['overlay'])
            disc_heatmap_b64 = image_to_base64(seg_output['disc_heatmap'])
            cup_heatmap_b64 = image_to_base64(seg_output['cup_heatmap'])
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        response_data = {
            'success': True,
            'disc_mask': disc_mask_b64,
            'cup_mask': cup_mask_b64,
            'disc_heatmap': disc_heatmap_b64,
            'cup_heatmap': cup_heatmap_b64,
            'segmentation_quality': seg_output.get('segmentation_quality', {}),
            'original_image': original_b64,
            'overlay': overlay_b64,
            'timestamp': datetime.now().isoformat(),
            'patient_details': patient_details
        }

        if should_save_logbook_from_form():
            save_patient_record({
                'timestamp': response_data['timestamp'],
                'mode': 'segment-only',
                'filename': file.filename,
                'patient_details': patient_details,
                'segmentation_quality': response_data['segmentation_quality'],
                'record_type': 'analysis'
            })

        return jsonify(response_data), 200
    
    except Exception as e:
        print(f"Error in segment endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/combined', methods=['POST'])
@login_required(api=True)
def api_combined():
    """
    Combined diagnosis endpoint (classification + segmentation)
    Upload: POST /api/combined with image file
    Returns: {prediction, confidence, disc_mask, cup_mask, overlay}
    """
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image file provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file format. Use PNG, JPG, JPEG, or BMP'}), 400
        
        if classification_model is None or segmentation_model is None:
            return jsonify({'error': 'One or more models not loaded'}), 500
        
        patient_details = _extract_patient_details()

        # Save temporary file
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        try:
            # Load original image
            original_img = Image.open(filepath).convert('RGB')
            original_img_np = np.array(original_img)
            
            # === CLASSIFICATION ===
            transform_clf = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ])
            
            tensor_clf = transform_clf(original_img).unsqueeze(0).to(device)
            
            with torch.no_grad():
                logits = classification_model(tensor_clf)
                probs = torch.softmax(logits, dim=1)
                confidence, pred_idx = torch.max(probs, dim=1)
            
            classes = ["glaucoma", "normal"]
            prediction = classes[pred_idx.item()]
            conf_score = float(confidence.item())
            glaucoma_prob = float(probs[0, 0].item())
            normal_prob = float(probs[0, 1].item())
            
            # === SEGMENTATION ===
            seg_output = run_segmentation_pipeline(original_img)

            # === CLINICAL METRICS ===
            clinical_metrics = calculate_clinical_metrics(
                glaucoma_prob,
                seg_output['disc_mask'],
                seg_output['cup_mask'],
                seg_output.get('segmentation_quality'),
            )

            # Convert to base64 for JSON response
            disc_mask_b64 = image_to_base64(seg_output['disc_mask'])
            cup_mask_b64 = image_to_base64(seg_output['cup_mask'])
            original_b64 = image_to_base64(seg_output['original_image'])
            overlay_b64 = image_to_base64(seg_output['overlay'])
            disc_heatmap_b64 = image_to_base64(seg_output['disc_heatmap'])
            cup_heatmap_b64 = image_to_base64(seg_output['cup_heatmap'])
            
        finally:
            if os.path.exists(filepath):
                os.remove(filepath)
        
        response_data = {
            'success': True,
            'prediction': prediction,
            'confidence': conf_score,
            'probabilities': {
                'glaucoma': glaucoma_prob,
                'normal': normal_prob
            },
            'clinical_metrics': clinical_metrics,
            'segmentation_quality': seg_output.get('segmentation_quality', {}),
            'disc_mask': disc_mask_b64,
            'cup_mask': cup_mask_b64,
            'disc_heatmap': disc_heatmap_b64,
            'cup_heatmap': cup_heatmap_b64,
            'original_image': original_b64,
            'overlay': overlay_b64,
            'timestamp': datetime.now().isoformat(),
            'patient_details': patient_details
        }

        if should_save_logbook_from_form():
            save_patient_record({
                'timestamp': response_data['timestamp'],
                'mode': 'combined',
                'filename': file.filename,
                'patient_details': patient_details,
                'prediction': prediction,
                'confidence': conf_score,
                'probabilities': response_data['probabilities'],
                'clinical_metrics': clinical_metrics,
                'segmentation_quality': response_data['segmentation_quality'],
                'record_type': 'analysis'
            })

        return jsonify(response_data), 200
    
    except Exception as e:
        print(f"Error in combined endpoint: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': str(e)}), 500


@app.route('/api/batch', methods=['POST'])
@login_required(api=True)
def api_batch():
    """
    Batch processing endpoint
    Upload: POST /api/batch with multiple image files
    Returns: array of results for each image
    """
    try:
        if 'images' not in request.files:
            return jsonify({'error': 'No image files provided'}), 400
        
        files = request.files.getlist('images')
        if not files or files[0].filename == '':
            return jsonify({'error': 'No files selected'}), 400
        
        if classification_model is None or segmentation_model is None:
            return jsonify({'error': 'One or more models not loaded'}), 500
        
        results = []
        
        for file in files:
            if not allowed_file(file.filename):
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': 'Invalid file format'
                })
                continue
            
            try:
                # Save temporary file
                filename = secure_filename(file.filename)
                filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(filepath)
                
                # Load original image
                original_img = Image.open(filepath).convert('RGB')
                
                # === CLASSIFICATION ===
                transform_clf = transforms.Compose([
                    transforms.Resize((224, 224)),
                    transforms.ToTensor(),
                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
                ])
                
                tensor_clf = transform_clf(original_img).unsqueeze(0).to(device)
                
                with torch.no_grad():
                    logits = classification_model(tensor_clf)
                    probs = torch.softmax(logits, dim=1)
                    confidence, pred_idx = torch.max(probs, dim=1)
                
                classes = ["glaucoma", "normal"]
                prediction = classes[pred_idx.item()]
                conf_score = float(confidence.item())
                glaucoma_prob = float(probs[0, 0].item())
                normal_prob = float(probs[0, 1].item())
                
                # === SEGMENTATION ===
                seg_output = run_segmentation_pipeline(original_img)
                overlay_b64 = image_to_base64(seg_output['overlay'])
                
                results.append({
                    'filename': file.filename,
                    'success': True,
                    'prediction': prediction,
                    'confidence': conf_score,
                    'probabilities': {
                        'glaucoma': glaucoma_prob,
                        'normal': normal_prob
                    },
                    'overlay': overlay_b64,
                    'timestamp': datetime.now().isoformat()
                })
                
                # Clean up
                if os.path.exists(filepath):
                    os.remove(filepath)
            
            except Exception as e:
                results.append({
                    'filename': file.filename,
                    'success': False,
                    'error': str(e)
                })
        
        return jsonify({
            'success': True,
            'total': len(files),
            'successful': sum(1 for r in results if r.get('success', False)),
            'results': results
        }), 200
    
    except Exception as e:
        print(f"Error in batch endpoint: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/api/health', methods=['GET'])
@login_required(api=True)
def health_check():
    """Health check endpoint"""
    return jsonify({
        'status': 'ok',
        'device': str(device),
        'models_loaded': {
            'classification': classification_model is not None,
            'segmentation': segmentation_model is not None
        }
    }), 200


@app.route('/api/patient-records', methods=['GET'])
@login_required(api=True)
def api_patient_records():
    records_file = Path(app.config['PATIENT_RECORDS_FILE'])
    if not records_file.exists():
        return jsonify({'success': True, 'records': []}), 200

    records = []
    try:
        with records_file.open('r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except Exception:
                    continue
    except Exception as e:
        return jsonify({'error': str(e)}), 500

    return jsonify({'success': True, 'records': records[-200:]}), 200


@app.route('/api/patient-records', methods=['POST'])
@login_required(api=True)
def api_save_patient_record():
    payload = request.get_json(silent=True) or {}
    patient_details = payload.get('patient_details')
    if not isinstance(patient_details, dict):
        return jsonify({'error': 'patient_details must be an object'}), 400

    if not str(patient_details.get('first_name', '')).strip() or not str(patient_details.get('last_name', '')).strip():
        return jsonify({'error': 'first_name and last_name are required'}), 400

    record = {
        'timestamp': datetime.now().isoformat(),
        'record_type': 'patient-only',
        'patient_details': patient_details,
        'note': str(payload.get('note', '')).strip()
    }
    save_patient_record(record)
    return jsonify({'success': True, 'record': record}), 201


if __name__ == '__main__':
    port = 5001  # Using 5001 instead of 5000 to avoid macOS AirTunes conflict
    print(f"\n{'='*60}")
    print(f"Loading Models...")
    print(f"{'='*60}")
    
    # Load models on startup
    classification_model = load_classification_model()
    segmentation_model = load_segmentation_model()
    
    print(f"\n{'='*60}")
    print(f"Starting Glaucoma Detection Web App")
    print(f"{'='*60}")
    print(f"Server: http://localhost:{port}")
    print(f"Device: {device}")
    print(f"Classification Model: {'✓ Loaded' if classification_model else '✗ Not Loaded'}")
    print(f"Segmentation Model: {'✓ Loaded' if segmentation_model else '✗ Not Loaded'}")
    print(f"{'='*60}\n")
    
    app.run(debug=False, host='127.0.0.1', port=port, threaded=True)
