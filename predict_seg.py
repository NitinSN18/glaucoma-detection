import argparse
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

from seg_model import build_seg_model


def detect_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def first_existing(paths: Sequence[str]) -> Optional[str]:
    for p in paths:
        if Path(p).exists():
            return p
    return None


def find_checkpoint(model_path: Optional[str]) -> str:
    if model_path and Path(model_path).exists():
        return model_path

    candidates = [
        "/kaggle/working/seg_model.pth",
        "/kaggle/input/models/nitinnarasipuram/seg-model-v1/pytorch/default/1/seg_model.pth",
        "/kaggle/input/models/nitinnarasipuram/seg-model-v1/pytorch/default/1/model.pth",
        "/kaggle/input/models/nitinnarasipuram/trained-model-final/pytorch/default/1/seg_model.pth",
        "/kaggle/input/models/nitinnarasipuram/trained-model-final/pytorch/default/1/model.pth",
        "seg_model.pth",
        "model.pth",
    ]

    for c in candidates:
        p = Path(c)
        if p.is_file():
            return str(p)

    for root in [
        "/kaggle/input/models/nitinnarasipuram/seg-model-v1/pytorch/default/1",
        "/kaggle/input/models/nitinnarasipuram/trained-model-final/pytorch/default/1",
        "/kaggle/working",
        ".",
    ]:
        r = Path(root)
        if not r.exists():
            continue
        for pat in ("*.pth", "*.pt", "*.bin"):
            files = list(r.glob(pat))
            if files:
                return str(files[0])

    raise FileNotFoundError("No segmentation checkpoint found. Pass --model-path.")


def choose_input_image(image_path: Optional[str]) -> str:
    if image_path and Path(image_path).exists():
        return image_path

    test_dirs = [
        "/kaggle/input/datasets/nitinnarasipuram/test-predict",
        "/kaggle/input/datasets/nitinnarasipuram/test-4",
        "/kaggle/input/datasets/nitinnarasipuram/test-3",
        "/kaggle/input/datasets/avinashreddy2309/glaucoma-detection/data/data/test",
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/REFUGE/test/Images",
        "/kaggle/input/datasets/arnavjain1/glaucoma-datasets/REFUGE/test/Images_Cropped",
    ]
    selected = first_existing(test_dirs)
    if selected:
        p = Path(selected)
        for ext in ("*.jpg", "*.jpeg", "*.png", "*.bmp", "*.tif", "*.tiff"):
            files = sorted(p.glob(ext))
            if files:
                return str(files[0])

    raise FileNotFoundError("No input test image found. Pass --image-path.")


def run(args: argparse.Namespace) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    device = detect_device()
    ckpt_path = find_checkpoint(args.model_path)
    input_image = choose_input_image(args.image_path)

    print(f"Device     : {device}")
    print(f"Checkpoint : {ckpt_path}")
    print(f"Image      : {input_image}")

    model = build_seg_model(num_classes=2)
    state = torch.load(ckpt_path, map_location=device)
    if isinstance(state, dict) and "state_dict" in state:
        state = state["state_dict"]
    if isinstance(state, dict):
        if any(k.startswith("module.") for k in state):
            state = {k.replace("module.", "", 1): v for k, v in state.items()}
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        raise RuntimeError(
            "Checkpoint is not compatible with current segmentation model. "
            f"Missing keys: {missing}. Unexpected keys: {unexpected}."
        )
    model.to(device).eval()

    image = Image.open(input_image).convert("RGB")
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5] * 3, [0.5] * 3),
    ])

    x = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(x)
        out = (torch.sigmoid(logits) > args.threshold).float().cpu().numpy()[0]

    disc = (out[0] * 255).astype(np.uint8)
    cup = (out[1] * 255).astype(np.uint8)
    disc = np.array(Image.fromarray(disc).resize((image.width, image.height), Image.NEAREST))
    cup = np.array(Image.fromarray(cup).resize((image.width, image.height), Image.NEAREST))

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    Image.fromarray(disc).save(out_dir / "disc_mask.png")
    Image.fromarray(cup).save(out_dir / "cup_mask.png")

    fig_path = out_dir / "prediction_overlay.png"
    plt.figure(figsize=(15, 5))
    plt.subplot(1, 3, 1)
    plt.imshow(image)
    plt.title("Original")
    plt.axis("off")

    plt.subplot(1, 3, 2)
    plt.imshow(image)
    plt.imshow(disc, alpha=0.4, cmap="Blues")
    plt.title("Disc")
    plt.axis("off")

    plt.subplot(1, 3, 3)
    plt.imshow(image)
    plt.imshow(cup, alpha=0.4, cmap="Reds")
    plt.title("Cup")
    plt.axis("off")

    plt.tight_layout()
    plt.savefig(fig_path, dpi=120)
    print(f"Saved outputs to: {out_dir}")


def parse_args() -> argparse.Namespace:
    default_output = "/kaggle/working" if Path("/kaggle/input").exists() else "."

    parser = argparse.ArgumentParser(description="Run segmentation prediction on one image")
    parser.add_argument("--model-path", type=str, default=None, help="Path to model checkpoint")
    parser.add_argument("--image-path", type=str, default=None, help="Path to input image")
    parser.add_argument("--output-dir", type=str, default=default_output, help="Directory for output masks/plot")
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument("--threshold", type=float, default=0.3)
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
