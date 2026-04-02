#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms

try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'efficientnet_pytorch'. Install it with: pip install efficientnet_pytorch"
    ) from exc


CLASSES = ["glaucoma", "normal"]
VALID_EXT = {".jpg", ".jpeg", ".png"}


def infer_efficientnet_name(state_dict: dict[str, torch.Tensor]) -> str:
    # Infer the model family from classifier input width in checkpoint.
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

    # Fallback if classifier width is unavailable: use stem width.
    if "_conv_stem.weight" in state_dict and state_dict["_conv_stem.weight"].ndim == 4:
        stem_out = int(state_dict["_conv_stem.weight"].shape[0])
        if stem_out == 32:
            return "efficientnet-b0"
        if stem_out == 48:
            return "efficientnet-b4"

    return "efficientnet-b4"


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def build_model(
    device: torch.device,
    model_path: Path | None = None,
    model_arch: str = "auto",
) -> nn.Module:
    model_path = model_path or resolve_model_path()
    state_dict = torch.load(str(model_path), map_location=device)

    selected_arch = infer_efficientnet_name(state_dict) if model_arch == "auto" else model_arch

    # Use from_name to avoid downloading pretrained weights at inference startup.
    model = EfficientNet.from_name(selected_arch)
    model._fc = nn.Linear(model._fc.in_features, 2)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    print(f"Using device: {device}")
    print(f"Model arch: {selected_arch}")
    print(f"Loaded model from: {model_path}")
    return model


def resolve_model_path(use_picker: bool = False) -> Path:
    candidates = [
        Path("best_model.pth"),
        Path(__file__).resolve().parent / "best_model.pth",
        Path("/kaggle/working/best_model.pth"),
    ]

    for path in candidates:
        if path.exists():
            return path

    if use_picker:
        selected = pick_model_file()
        if selected is not None:
            return selected

    raise FileNotFoundError(
        "best_model.pth not found. Place it in the project root or pass --model."
    )


def resolve_image_paths(cli_paths: Iterable[str], use_picker: bool) -> list[Path]:
    resolved: list[Path] = []

    for p in cli_paths:
        path = Path(p).expanduser().resolve()
        if path.is_file() and path.suffix.lower() in VALID_EXT:
            resolved.append(path)

    if resolved:
        return resolved

    if Path("/kaggle/input").exists():
        kaggle_test = Path(
            "/kaggle/input/datasets/avinashreddy2309/glaucoma-detection/data/data/test"
        )
        if kaggle_test.exists():
            return sorted(
                [
                    p
                    for p in kaggle_test.iterdir()
                    if p.is_file() and p.suffix.lower() in VALID_EXT
                ]
            )

    if use_picker:
        return pick_images()

    return []


def _create_hidden_tk_root():
    try:
        import tkinter as tk
    except Exception:
        return None, None

    root = tk.Tk()
    root.withdraw()
    return tk, root


def pick_images() -> list[Path]:
    tk, root = _create_hidden_tk_root()
    if root is None:
        return []

    from tkinter import filedialog

    chosen = filedialog.askopenfilenames(
        title="Select Fundus Image(s)",
        filetypes=[("Image Files", "*.jpg *.jpeg *.png")],
    )
    root.destroy()

    if not chosen:
        return []

    image_paths: list[Path] = []
    for p in chosen:
        path = Path(p).expanduser().resolve()
        if path.is_file() and path.suffix.lower() in VALID_EXT:
            image_paths.append(path)

    return image_paths


def pick_model_file() -> Path | None:
    _, root = _create_hidden_tk_root()
    if root is None:
        return None

    from tkinter import filedialog

    chosen = filedialog.askopenfilename(
        title="Select Model File (.pth)",
        filetypes=[("PyTorch Model", "*.pth"), ("All Files", "*.*")],
    )
    root.destroy()

    if not chosen:
        return None

    return Path(chosen).expanduser().resolve()


def predict_one(model: nn.Module, image_path: Path, device: torch.device) -> tuple[str, float]:
    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    image = Image.open(image_path).convert("RGB")
    tensor = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)
        confidence, pred_idx = torch.max(probs, dim=1)

    return CLASSES[pred_idx.item()], float(confidence.item())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Glaucoma classifier inference")
    parser.add_argument("images", nargs="*", help="One or more image paths")
    parser.add_argument(
        "--no-picker",
        action="store_true",
        help="Disable macOS file picker fallback and require CLI image path(s)",
    )
    parser.add_argument(
        "--model",
        default=None,
        help="Optional explicit path to model .pth file",
    )
    parser.add_argument(
        "--pick-model",
        action="store_true",
        help="Open file picker to choose model .pth if default model is missing",
    )
    parser.add_argument(
        "--model-arch",
        default="auto",
        choices=[
            "auto",
            "efficientnet-b0",
            "efficientnet-b1",
            "efficientnet-b2",
            "efficientnet-b3",
            "efficientnet-b4",
            "efficientnet-b5",
            "efficientnet-b6",
            "efficientnet-b7",
        ],
        help="Model architecture to build before loading checkpoint",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    device = get_device()

    if args.model:
        model_path = Path(args.model).expanduser().resolve()
        if not model_path.exists():
            print(f"Model file not found: {model_path}")
            return 1
    else:
        try:
            model_path = resolve_model_path(use_picker=args.pick_model)
        except FileNotFoundError:
            if not args.pick_model:
                print("Default model not found. Re-run with --pick-model to choose a .pth file.")
                return 1
            raise

    try:
        model = build_model(device, model_path=model_path, model_arch=args.model_arch)
    except RuntimeError as exc:
        print(f"Failed to load checkpoint: {exc}")
        print("Try overriding architecture, for example: --model-arch efficientnet-b0")
        return 1

    image_paths = resolve_image_paths(args.images, use_picker=not args.no_picker)
    if not image_paths:
        print("No valid images provided.")
        print("Usage: python predict.py <image_path1> [image_path2 ...]")
        print("Tip: run without --no-picker to choose a file in a GUI dialog on macOS.")
        return 1

    for image_path in image_paths:
        try:
            label, confidence = predict_one(model, image_path, device)
            print(f"\nImage     : {image_path}")
            print(f"Prediction: {label}")
            print(f"Confidence: {confidence:.4f}")
        except Exception as exc:
            print(f"Error processing {image_path}: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
