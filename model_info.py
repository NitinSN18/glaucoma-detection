#!/usr/bin/env python3
from __future__ import annotations

import torch
import torch.nn as nn

try:
    from efficientnet_pytorch import EfficientNet
except ModuleNotFoundError as exc:
    raise ModuleNotFoundError(
        "Missing dependency 'efficientnet_pytorch'. Install it with: pip install -r requirements.txt"
    ) from exc


def build_classification_model() -> nn.Module:
    model = EfficientNet.from_name("efficientnet-b4")
    model._fc = nn.Linear(model._fc.in_features, 2)
    return model


def leaf_modules(model: nn.Module) -> list[tuple[str, nn.Module]]:
    return [(name, module) for name, module in model.named_modules() if name and not list(module.children())]


def dense_layers(model: nn.Module) -> list[tuple[str, nn.Linear]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, nn.Linear)]


def main() -> int:
    model = build_classification_model()

    include_top = getattr(getattr(model, "_global_params", None), "include_top", None)
    modules_excluding_root = len(list(model.modules())) - 1
    leaves = leaf_modules(model)
    dense = dense_layers(model)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print("EfficientNet-B4 classification model report")
    print("-" * 48)
    print("Base constructor in training code: EfficientNet.from_pretrained('efficientnet-b4')")
    print("Utility constructor (no weight download): EfficientNet.from_name('efficientnet-b4')")
    print(f"include_top (EfficientNet global params): {include_top}")
    print("Added/modified head in this repo: model._fc = nn.Linear(model._fc.in_features, 2)")
    print()
    print("Layer counting definitions:")
    print(f"- modules excluding root (len(list(model.modules())) - 1): {modules_excluding_root}")
    print(f"- leaf layers (modules with no children): {len(leaves)}")
    print()
    print("Dense layers (neurons = out_features):")
    for name, layer in dense:
        print(f"- {name}: in_features={layer.in_features}, out_features={layer.out_features}")
    print()
    print("Parameter totals:")
    print(f"- total parameters: {total_params:,}")
    print(f"- trainable parameters: {trainable_params:,}")
    print("Formula: Linear params = in_features * out_features + out_features (bias)")
    print()
    print("Raw model print:")
    print(model)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
