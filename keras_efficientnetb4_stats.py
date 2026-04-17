#!/usr/bin/env python3
from __future__ import annotations

import argparse

import tensorflow as tf
from tensorflow import keras


def build_keras_model(input_size: int, num_classes: int) -> tuple[keras.Model, keras.Model]:
    """Build EfficientNet-B4 classifier and return (full_model, backbone_model).

    Args:
        input_size: Square input image size (e.g., 224 for 224x224x3).
        num_classes: Number of output classes for the final Dense layer.

    Returns:
        Tuple of:
            - full classifier model (EfficientNet-B4 + GAP + Dense head)
            - EfficientNet-B4 backbone submodel
    """
    inputs = keras.Input(shape=(input_size, input_size, 3), name="input_image")
    base_model = keras.applications.EfficientNetB4(
        include_top=False,
        weights=None,
        name="efficientnetb4_backbone",
    )
    x = base_model(inputs, training=False)
    x = keras.layers.GlobalAveragePooling2D(name="avg_pool")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax", name="classifier")(x)
    model = keras.Model(inputs, outputs, name="glaucoma_efficientnetb4_classifier")
    return model, base_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print Keras EfficientNet-B4 layer count, Dense units, and parameter counts."
    )
    parser.add_argument("--input-size", type=int, default=224, help="Input image size (square)")
    parser.add_argument("--num-classes", type=int, default=2, help="Number of output classes")
    parser.add_argument(
        "--expand-nested-summary",
        action="store_true",
        help="Show expanded nested summary for internal EfficientNet layers",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model, base_model = build_keras_model(args.input_size, args.num_classes)

    print("=== model.summary() ===")
    model.summary(expand_nested=args.expand_nested_summary)

    dense_units = [layer.units for layer in model.layers if isinstance(layer, keras.layers.Dense)]
    top_level_layers = len(model.layers)
    efficientnet_internal_layers = len(base_model.layers)
    expanded_total_layers = efficientnet_internal_layers + 2  # GAP + Dense; Input is already in base.

    print("\n=== Computed counts ===")
    print(f"Top-level Keras model layers (len(model.layers)): {top_level_layers}")
    print(
        "EfficientNetB4 submodel internal layers "
        f"(len(base_model.layers)): {efficientnet_internal_layers}"
    )
    print(
        "Expanded total layers (EfficientNet internal + classifier head): "
        f"{expanded_total_layers}"
    )
    print(f"Dense layer units (neurons): {dense_units}")
    print(f"Total params: {model.count_params()}")
    print(f"Trainable params: {sum(int(tf.size(v)) for v in model.trainable_weights)}")
    print(f"Non-trainable params: {sum(int(tf.size(v)) for v in model.non_trainable_weights)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
