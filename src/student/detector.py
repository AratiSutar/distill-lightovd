"""
Full student detector: lightweight backbone + FCOS-style detection head.
This is the complete model that will be trained via distillation from the
OWL-ViT teacher.
"""

import torch.nn as nn

from src.student.backbone import LightBackbone
from src.student.head import DetectionHead


class StudentDetector(nn.Module):
    """
    End-to-end lightweight object detector.
    Input: (batch, 3, 320, 320) image
    Output: cls_logits, reg_preds, centerness (all at stride-16 resolution, 20x20)
    """

    def __init__(self, num_classes=12):
        super().__init__()
        self.backbone = LightBackbone()
        self.head = DetectionHead(in_channels=256, num_classes=num_classes)

    def forward(self, x):
        features = self.backbone(x)
        cls_logits, reg_preds, centerness = self.head(features)
        return cls_logits, reg_preds, centerness
