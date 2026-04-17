#!/usr/bin/env python3
from __future__ import annotations

import torch.nn as nn

try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'efficientnet_pytorch'. Install it with: pip install -r requirements.txt"
    ) from exc


MODEL_NAME = "efficientnet-b4"


def build_classification_model() -> nn.Module:
    # Mirrors train.py architecture while avoiding pretrained weight download.
    model = EfficientNet.from_name(MODEL_NAME)
    model._fc = nn.Linear(model._fc.in_features, 2)
    return model


def main() -> int:
    base_model = EfficientNet.from_name(MODEL_NAME)
    include_top_equivalent = isinstance(base_model._fc, nn.Linear)
    original_output_units = base_model._fc.out_features

    model = build_classification_model()

    top_level_layers = len(list(model.children()))
    efficientnet_internal_modules = sum(1 for _ in model.modules())
    linear_layers_info = [(name, layer.out_features) for name, layer in model.named_modules() if isinstance(layer, nn.Linear)]
    output_units = model._fc.out_features

    print("=== EfficientNet-B4 classification model inspection ===")
    print("Framework: PyTorch (efficientnet_pytorch)")
    print(f"EfficientNet include_top equivalent: {include_top_equivalent} (default output units={original_output_units})")
    print("\nModel summary:")
    print(model)
    print(f"\nTotal top-level model layers (len(list(model.children()))): {top_level_layers}")
    print(f"EfficientNet-B4 internal module count (sum(1 for _ in model.modules())): {efficientnet_internal_modules}")
    linear_layers_text = ", ".join(f"{name}={units}" for name, units in linear_layers_info) or "None"
    print(f"Head Linear layers and units: {linear_layers_text}")
    print(f"Output layer units (num_classes): {output_units}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
