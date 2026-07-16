"""
FCOS-style anchor-free detection head for the student model.
Operates on a single-scale feature map from the backbone.
"""

import torch.nn as nn


class DetectionHead(nn.Module):
    """
    Predicts, for every spatial location in the feature map:
      - class scores (num_classes)
      - box regression: distances to (left, top, right, bottom) edges
      - centerness: how close this location is to a real object center
    """

    def __init__(self, in_channels=256, num_classes=12, head_channels=128):
        super().__init__()

        # Shared conv tower before splitting into branches
        self.shared_conv = nn.Sequential(
            nn.Conv2d(in_channels, head_channels, 3, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(head_channels, head_channels, 3, padding=1),
            nn.BatchNorm2d(head_channels),
            nn.ReLU(inplace=True),
        )

        # Classification branch: predicts class scores per location
        self.cls_head = nn.Conv2d(head_channels, num_classes, 3, padding=1)

        # Regression branch: predicts (left, top, right, bottom) distances
        self.reg_head = nn.Conv2d(head_channels, 4, 3, padding=1)

        # Centerness branch: predicts a single score per location
        self.centerness_head = nn.Conv2d(head_channels, 1, 3, padding=1)

    def forward(self, features):
        x = self.shared_conv(features)

        cls_logits = self.cls_head(x)  # (batch, num_classes, H, W)
        reg_preds = self.reg_head(x)  # (batch, 4, H, W)
        centerness = self.centerness_head(x)  # (batch, 1, H, W)

        return cls_logits, reg_preds, centerness
