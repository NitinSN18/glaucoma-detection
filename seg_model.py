import torch
import torch.nn as nn


class UNet(nn.Module):
    """Minimal two-level U-Net for optic disc/cup segmentation.

    Outputs two channels (channel 0: disc mask, channel 1: cup mask).
    """

    def __init__(self, out_channels: int = 2):
        super().__init__()
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.pool = nn.MaxPool2d(2)
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(),
        )
        self.up = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
        )
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x1 = self.enc1(x)
        x2 = self.enc2(self.pool(x1))
        x3 = self.up(x2)
        x4 = torch.cat([x3, x1], dim=1)
        return self.final(self.dec(x4))
