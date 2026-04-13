import torch
import torch.nn as nn
from torchvision.models import ResNet50_Weights
from torchvision.models.segmentation import deeplabv3_resnet50


class DeepLabSegModel(nn.Module):
    def __init__(self, num_classes: int = 2, pretrained_backbone: bool = False):
        super().__init__()
        self.model = deeplabv3_resnet50(
            weights=None,
            weights_backbone=ResNet50_Weights.DEFAULT if pretrained_backbone else None,
            num_classes=num_classes,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)["out"]


def build_seg_model(num_classes: int = 2, pretrained_backbone: bool = False) -> nn.Module:
    return DeepLabSegModel(num_classes=num_classes, pretrained_backbone=pretrained_backbone)
