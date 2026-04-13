import torch
import torch.nn as nn
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabSegModel(nn.Module):
    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=None,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def build_seg_model(num_classes: int = 2) -> nn.Module:
    return DeepLabSegModel(num_classes=num_classes)
