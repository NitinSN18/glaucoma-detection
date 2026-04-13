#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

from seg_model import create_segmentation_model

try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'efficientnet_pytorch'. Install it with: pip install -r requirements.txt"
    ) from exc


CLASSES = ["glaucoma", "normal"]
VALID_EXT = {".jpg", ".jpeg", ".png"}


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _create_hidden_tk_root():
    try:
        import tkinter as tk
    except Exception:
        return None

    root = tk.Tk()
    root.withdraw()
    return root


def pick_single_image() -> Path | None:
    root = _create_hidden_tk_root()
    if root is None:
        return None

    from tkinter import filedialog

    chosen = filedialog.askopenfilename(
        title="Select Fundus Image",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")],
    )
    root.destroy()

    if not chosen:
        return None

    p = Path(chosen).expanduser().resolve()
    if p.is_file() and p.suffix.lower() in VALID_EXT:
        return p
    return None


def infer_efficientnet_name(state_dict: dict[str, torch.Tensor]) -> str:
    if "_fc.weight" in state_dict and state_dict["_fc.weight"].ndim == 2:
        in_features = int(state_dict["_fc.weight"].shape[1])
        by_fc_width = {
            1280: "efficientnet-b0",
            1408: "efficientnet-b2",
            1536: "efficientnet-b3",
            1792: "efficientnet-b4",
            2048: "efficientnet-b5",
            2304: "efficientnet-b6",
            2560: "efficientnet-b7",
        }
        if in_features in by_fc_width:
            return by_fc_width[in_features]
    return "efficientnet-b4"


def load_classifier(model_path: Path, device: torch.device, model_arch: str = "auto") -> nn.Module:
    state_dict = torch.load(str(model_path), map_location=device)
    selected_arch = infer_efficientnet_name(state_dict) if model_arch == "auto" else model_arch

    model = EfficientNet.from_name(selected_arch)
    model._fc = nn.Linear(model._fc.in_features, 2)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    print(f"[DEBUG] Classifier arch: {selected_arch}")
    print(f"[DEBUG] Classifier ckpt: {model_path}")
    return model


def predict_classification(model: nn.Module, image_pil: Image.Image, device: torch.device) -> tuple[str, float, float]:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    tensor = transform(image_pil).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)

    glaucoma_prob = float(probs[0, 0].item())
    confidence, pred_idx = torch.max(probs, dim=1)
    return CLASSES[pred_idx.item()], float(confidence.item()), glaucoma_prob


def _fullframe_inner_ellipse_mask(shape_hw: tuple[int, int]) -> np.ndarray:
    h, w = shape_hw
    full_mask = np.zeros((h, w), dtype=np.uint8)
    center = (w // 2, h // 2)
    axes = (int(0.47 * w), int(0.47 * h))
    cv2.ellipse(full_mask, center, axes, 0, 0, 360, 255, -1)
    return full_mask


def get_fundus_mask(rgb_img: np.ndarray, roi_mode: str = "auto") -> tuple[np.ndarray, str]:
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


def crop_to_mask(rgb_img: np.ndarray, bin_mask: np.ndarray, margin: float = 0.05):
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


def build_transform(size: int):
    return transforms.Compose([transforms.Resize((size, size)), transforms.ToTensor()])


def predict_with_tta(seg_model: nn.Module, crop_pil: Image.Image, model_device: torch.device, input_size: int):
    tform = build_transform(input_size)
    x0 = tform(crop_pil).unsqueeze(0).to(model_device)
    variants = [(x0, None), (torch.flip(x0, dims=[3]), "h"), (torch.flip(x0, dims=[2]), "v")]

    pred_sum = None
    with torch.no_grad():
        for inp, tag in variants:
            out = torch.sigmoid(seg_model(inp))
            if tag == "h":
                out = torch.flip(out, dims=[3])
            elif tag == "v":
                out = torch.flip(out, dims=[2])
            pred_sum = out if pred_sum is None else pred_sum + out

    return (pred_sum / len(variants)).squeeze(0).cpu().numpy()


def choose_component_near_point(bin_img: np.ndarray, point_xy: tuple[int, int], min_area: int = 25) -> np.ndarray:
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
        score = dist2 - 0.02 * area
        if score < best_score:
            best_score = score
            best = c

    if best is not None:
        cv2.drawContours(out, [best], -1, 255, -1)
    return out


def build_disc_prior(rgb_img: np.ndarray, valid_mask: np.ndarray):
    rgb = rgb_img.astype(np.float32)
    score = 0.6 * rgb[:, :, 0] + 0.4 * rgb[:, :, 1] - 0.2 * rgb[:, :, 2]
    score = np.where(valid_mask > 0, score, 0.0)

    vals = score[valid_mask > 0]
    if vals.size == 0:
        h, w = valid_mask.shape
        return np.ones((h, w), dtype=np.float32), (w // 2, h // 2)

    p2, p98 = np.percentile(vals, [2, 98])
    score = np.clip((score - p2) / max(1e-6, p98 - p2), 0.0, 1.0)
    score = cv2.GaussianBlur(score, (7, 7), 0)
    masked = np.where(valid_mask > 0, score, 0.0)
    y, x = np.unravel_index(np.argmax(masked), masked.shape)

    h, w = valid_mask.shape
    yy, xx = np.mgrid[0:h, 0:w]
    sigma_prior = max(12.0, min(h, w) * 0.16)
    dist2 = (xx - x) ** 2 + (yy - y) ** 2
    prior = np.exp(-dist2 / (2.0 * sigma_prior * sigma_prior)).astype(np.float32)
    prior = np.where(valid_mask > 0, prior, 0.0)
    return prior, (x, y)


def adaptive_binary(prob_map: np.ndarray, valid_mask: np.ndarray, pct: float, min_thr: float, max_thr: float):
    vals = prob_map[valid_mask > 0]
    if vals.size == 0:
        return np.zeros_like(prob_map, dtype=np.uint8), min_thr

    thr = float(np.percentile(vals, pct))
    thr = max(min_thr, min(max_thr, thr))
    return (prob_map >= thr).astype(np.uint8) * 255, thr


def load_segmentation_model(arch: str, encoder: str, model_path: Path, device: torch.device) -> nn.Module:
    seg_model = create_segmentation_model(
        arch=arch,
        out_channels=2,
        encoder_name=encoder,
        encoder_weights=None,
    )
    ckpt = torch.load(str(model_path), map_location=torch.device("cpu"))
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:
            ckpt = ckpt["state_dict"]
        elif "model_state_dict" in ckpt:
            ckpt = ckpt["model_state_dict"]
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}

    seg_model.load_state_dict(ckpt)
    seg_model.to(device)
    seg_model.eval()

    print(f"[DEBUG] Segmentation arch: {arch}")
    print(f"[DEBUG] Segmentation encoder: {encoder}")
    print(f"[DEBUG] Segmentation ckpt: {model_path}")
    return seg_model


def segment_and_extract(seg_model: nn.Module, image_pil: Image.Image, device: torch.device, roi_mode: str = "auto"):
    image_np = np.array(image_pil.convert("RGB"))
    h0, w0 = image_np.shape[:2]

    fundus_mask_full, mode_used = get_fundus_mask(image_np, roi_mode=roi_mode)
    crop_np, (x1, y1, x2, y2) = crop_to_mask(image_np, fundus_mask_full)
    crop_pil = Image.fromarray(crop_np)

    input_size = 256
    scales = (224, 256, 320)
    pred_acc = None
    for size in scales:
        p = predict_with_tta(seg_model, crop_pil, device, input_size=size)
        if p.shape[1] != input_size or p.shape[2] != input_size:
            p = np.stack(
                [
                    cv2.resize(p[0], (input_size, input_size), interpolation=cv2.INTER_LINEAR),
                    cv2.resize(p[1], (input_size, input_size), interpolation=cv2.INTER_LINEAR),
                ],
                axis=0,
            )
        pred_acc = p if pred_acc is None else pred_acc + p
    crop_prob = pred_acc / float(len(scales))

    crop_h, crop_w = crop_np.shape[:2]
    disc_crop_prob = cv2.resize(crop_prob[0], (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)
    cup_crop_prob = cv2.resize(crop_prob[1], (crop_w, crop_h), interpolation=cv2.INTER_LINEAR)

    disc_prob = np.zeros((h0, w0), dtype=np.float32)
    cup_prob = np.zeros((h0, w0), dtype=np.float32)
    disc_prob[y1:y2, x1:x2] = disc_crop_prob
    cup_prob[y1:y2, x1:x2] = cup_crop_prob

    valid_mask = fundus_mask_full
    disc_prob = disc_prob * (valid_mask / 255.0)
    cup_prob = cup_prob * (valid_mask / 255.0)

    disc_prior, (prior_x, prior_y) = build_disc_prior(image_np, valid_mask)
    model_peak = float(disc_prob.max())
    prior_weight = 0.60 if model_peak < 0.45 else 0.35
    gating_disc = np.clip(0.40 + 0.60 * disc_prior, 0.0, 1.0)
    gating_cup = np.clip(0.25 + 0.75 * disc_prior, 0.0, 1.0)

    disc_prob = (1.0 - prior_weight) * disc_prob + prior_weight * (disc_prob * gating_disc)
    cup_prob = (1.0 - prior_weight) * cup_prob + prior_weight * (cup_prob * gating_cup)

    disc_min_thr = 0.08 if float(disc_prob.max()) < 0.25 else 0.16
    cup_min_thr = 0.08 if float(cup_prob.max()) < 0.25 else 0.16

    disc_mask_raw, disc_thr = adaptive_binary(disc_prob, valid_mask, pct=97.2, min_thr=disc_min_thr, max_thr=0.75)
    cup_mask_raw, cup_thr = adaptive_binary(cup_prob, valid_mask, pct=98.5, min_thr=cup_min_thr, max_thr=0.82)

    kernel_d = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (9, 9))
    kernel_c = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    disc_mask = cv2.morphologyEx(disc_mask_raw, cv2.MORPH_CLOSE, kernel_d)
    disc_mask = cv2.morphologyEx(disc_mask, cv2.MORPH_OPEN, kernel_c)
    disc_mask = choose_component_near_point(disc_mask, (prior_x, prior_y), min_area=max(60, int(0.0008 * h0 * w0)))
    disc_mask = np.where(valid_mask > 0, disc_mask, 0).astype(np.uint8)

    cup_mask = cv2.morphologyEx(cup_mask_raw, cv2.MORPH_CLOSE, kernel_c)
    cup_mask = cv2.morphologyEx(cup_mask, cv2.MORPH_OPEN, kernel_c)
    cup_mask = np.where(disc_mask > 0, cup_mask, 0).astype(np.uint8)
    cup_mask = choose_component_near_point(cup_mask, (prior_x, prior_y), min_area=max(25, int(0.00025 * h0 * w0)))
    cup_mask = np.where(disc_mask > 0, cup_mask, 0).astype(np.uint8)

    disc_area = int(np.sum(disc_mask > 0))
    cup_area = int(np.sum(cup_mask > 0))
    cdr = (cup_area / max(1, disc_area)) if disc_area > 0 else 0.0

    return {
        "image_np": image_np,
        "disc_prob": disc_prob,
        "cup_prob": cup_prob,
        "disc_mask": disc_mask,
        "cup_mask": cup_mask,
        "disc_area": disc_area,
        "cup_area": cup_area,
        "cdr": float(cdr),
        "roi_mode_used": mode_used,
        "disc_thr": float(disc_thr),
        "cup_thr": float(cup_thr),
    }


def cdr_rule_label(cdr: float) -> str:
    if cdr < 0.3:
        return "LOW"
    if cdr < 0.6:
        return "MODERATE"
    return "HIGH"


def fused_risk_score(class_glaucoma_prob: float, cdr: float) -> float:
    # Normalize CDR into [0,1] where 0.3=0 risk contribution and 0.7~=1.
    cdr_norm = np.clip((cdr - 0.30) / 0.40, 0.0, 1.0)
    return float(np.clip(0.65 * class_glaucoma_prob + 0.35 * cdr_norm, 0.0, 1.0))


def fused_label(score: float) -> str:
    if score < 0.35:
        return "normal"
    if score < 0.65:
        return "suspect"
    return "glaucoma"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combined glaucoma inference (segmentation + classification)")
    parser.add_argument("image", nargs="?", help="Path to image. If omitted, opens picker.")
    parser.add_argument("--no-picker", action="store_true", help="Disable file picker fallback")

    parser.add_argument("--class-model", default="best_model.pth", help="Path to classifier .pth")
    parser.add_argument(
        "--class-arch",
        default="auto",
        choices=["auto", "efficientnet-b0", "efficientnet-b1", "efficientnet-b2", "efficientnet-b3", "efficientnet-b4", "efficientnet-b5", "efficientnet-b6", "efficientnet-b7"],
        help="Classifier architecture",
    )

    parser.add_argument("--seg-arch", default=os.getenv("SEG_MODEL_ARCH", "deeplabv3plus"))
    parser.add_argument("--seg-encoder", default=os.getenv("SEG_MODEL_ENCODER", "resnet34"))
    parser.add_argument(
        "--seg-model",
        default=os.getenv("SEG_MODEL_PATH", "seg_model_deeplabv3plus.pth"),
        help="Path to segmentation checkpoint",
    )
    parser.add_argument("--roi-mode", default="auto", choices=["auto", "blackbg", "fullframe"])
    parser.add_argument("--show-plots", action="store_true", help="Show segmentation overlays and heatmaps")
    return parser.parse_args()


def resolve_image_arg(image_arg: str | None, allow_picker: bool) -> Path | None:
    if image_arg:
        p = Path(image_arg).expanduser().resolve()
        if p.is_file() and p.suffix.lower() in VALID_EXT:
            return p
        return None

    if allow_picker:
        return pick_single_image()
    return None


def main() -> int:
    args = parse_args()
    device = get_device()

    image_path = resolve_image_arg(args.image, allow_picker=not args.no_picker)
    if image_path is None:
        print("No valid image selected/provided.")
        return 1

    class_model_path = Path(args.class_model).expanduser().resolve()
    seg_model_path = Path(args.seg_model).expanduser().resolve()

    if not class_model_path.exists():
        print(f"Classifier model not found: {class_model_path}")
        return 1
    if not seg_model_path.exists():
        print(f"Segmentation model not found: {seg_model_path}")
        return 1

    print(f"[DEBUG] Device: {device}")
    print(f"[DEBUG] Image: {image_path}")

    classifier = load_classifier(class_model_path, device, model_arch=args.class_arch)
    seg_model = load_segmentation_model(args.seg_arch, args.seg_encoder, seg_model_path, device)

    image_pil = Image.open(image_path).convert("RGB")

    class_label, class_conf, class_gl_prob = predict_classification(classifier, image_pil, device)
    seg = segment_and_extract(seg_model, image_pil, device, roi_mode=args.roi_mode)

    rule = cdr_rule_label(seg["cdr"])
    fused = fused_risk_score(class_gl_prob, seg["cdr"])
    final_label = fused_label(fused)

    print("\n" + "=" * 60)
    print("COMBINED GLAUCOMA INFERENCE")
    print("=" * 60)
    print(f"Image                 : {image_path}")
    print(f"Classifier prediction : {class_label} (confidence: {class_conf:.4f})")
    print(f"Classifier P(glaucoma): {class_gl_prob:.4f}")
    print(f"CDR                   : {seg['cdr']:.4f}")
    print(f"CDR rule risk         : {rule}")
    print(f"Disc area / Cup area  : {seg['disc_area']} / {seg['cup_area']}")
    print(f"Fused risk score      : {fused:.4f}")
    print(f"Final fused label     : {final_label}")
    print("=" * 60)

    if args.show_plots:
        image_np = seg["image_np"]

        plt.figure(figsize=(16, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.imshow(image_np)
        plt.imshow(seg["disc_prob"], cmap="hot", alpha=0.6, vmin=0.0, vmax=1.0)
        plt.title("Disc Probability")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.imshow(image_np)
        plt.imshow(seg["cup_prob"], cmap="hot", alpha=0.6, vmin=0.0, vmax=1.0)
        plt.title("Cup Probability")
        plt.axis("off")
        plt.tight_layout()
        plt.show(block=False)

        plt.figure(figsize=(16, 5))
        plt.subplot(1, 3, 1)
        plt.imshow(image_np)
        plt.title("Original")
        plt.axis("off")

        plt.subplot(1, 3, 2)
        disc_overlay = np.zeros_like(image_np)
        disc_overlay[seg["disc_mask"] > 0] = [0, 100, 255]
        plt.imshow(image_np)
        plt.imshow(disc_overlay, alpha=0.5)
        plt.title("Disc Mask")
        plt.axis("off")

        plt.subplot(1, 3, 3)
        cup_overlay = np.zeros_like(image_np)
        cup_overlay[seg["cup_mask"] > 0] = [255, 100, 0]
        plt.imshow(image_np)
        plt.imshow(cup_overlay, alpha=0.5)
        plt.title(f"Cup Mask (CDR: {seg['cdr']:.4f})")
        plt.axis("off")

        plt.tight_layout()
        plt.show()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
