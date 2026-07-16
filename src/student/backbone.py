"""
Lightweight CNN backbone for the student detector.
Designed for fast CPU inference on edge devices.
"""


import torch.nn as nn


class ConvBlock(nn.Module):
    """Conv -> BatchNorm -> ReLU, the basic building block."""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=False,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class LightBackbone(nn.Module):
    """
    Small CNN backbone: input 320x320x3 -> output 20x20x256 (stride 16).
    ~4 downsampling stages.
    """

    def __init__(self):
        super().__init__()
        self.stage1 = nn.Sequential(
            ConvBlock(3, 32, stride=2),  # 320 -> 160
            ConvBlock(32, 32, stride=1),
        )
        self.stage2 = nn.Sequential(
            ConvBlock(32, 64, stride=2),  # 160 -> 80
            ConvBlock(64, 64, stride=1),
        )
        self.stage3 = nn.Sequential(
            ConvBlock(64, 128, stride=2),  # 80 -> 40
            ConvBlock(128, 128, stride=1),
        )
        self.stage4 = nn.Sequential(
            ConvBlock(128, 256, stride=2),  # 40 -> 20
            ConvBlock(256, 256, stride=1),
        )

    def forward(self, x):
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.stage4(x)
        return x  # shape: (batch, 256, 20, 20)
