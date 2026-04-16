import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, out_channels=2):
        super(UNet, self).__init__()

        # Encoder - Level 1
        self.enc1 = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        self.pool = nn.MaxPool2d(2)

        # Encoder - Level 2
        self.enc2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Encoder - Level 3
        self.enc3 = nn.Sequential(
            nn.Conv2d(128, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Bottleneck
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True)
        )

        # Decoder - Level 3
        self.up3 = nn.ConvTranspose2d(512, 256, 2, stride=2)
        self.dec3 = nn.Sequential(
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True)
        )

        # Decoder - Level 2
        self.up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
        self.dec2 = nn.Sequential(
            nn.Conv2d(256, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True)
        )

        # Decoder - Level 1
        self.up1 = nn.ConvTranspose2d(128, 64, 2, stride=2)
        self.dec1 = nn.Sequential(
            nn.Conv2d(128, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True)
        )

        # Final output layer
        self.final = nn.Conv2d(64, out_channels, 1)

    def forward(self, x):
        # Encoder
        x1 = self.enc1(x)  # 256x256x64
        x2 = self.enc2(self.pool(x1))  # 128x128x128
        x3 = self.enc3(self.pool(x2))  # 64x64x256
        
        # Bottleneck
        b = self.bottleneck(self.pool(x3))  # 32x32x512

        # Decoder
        y3 = self.dec3(torch.cat([self.up3(b), x3], dim=1))  # 64x64x256
        y2 = self.dec2(torch.cat([self.up2(y3), x2], dim=1))  # 128x128x128
        y1 = self.dec1(torch.cat([self.up1(y2), x1], dim=1))  # 256x256x64

        return self.final(y1)


def create_segmentation_model(
    arch="unet",
    out_channels=2,
    encoder_name="resnet34",
    encoder_weights="imagenet",
):
    arch_norm = str(arch).strip().lower()

    if arch_norm == "unet":
        return UNet(out_channels=out_channels)

    if arch_norm in {"deeplabv3+", "deeplabv3plus"}:
        try:
            import segmentation_models_pytorch as smp
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "DeepLabV3+ requires segmentation-models-pytorch. "
                "Install dependencies with: pip install -r requirements.txt"
            ) from exc

        return smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=3,
            classes=out_channels,
            activation=None,
        )

    raise ValueError(
        f"Unsupported segmentation architecture: {arch}. "
        "Use one of: unet, deeplabv3plus"
    )
