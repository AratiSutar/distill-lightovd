"""
Tests for the student detection head architecture.
"""

import torch


def test_detection_head_output_shapes():
    """Confirm the head produces correctly shaped outputs for a dummy backbone feature map."""
    from src.student.head import DetectionHead

    model = DetectionHead(in_channels=256, num_classes=12)
    dummy_features = torch.randn(1, 256, 20, 20)  # matches backbone output shape

    cls_logits, reg_preds, centerness = model(dummy_features)

    assert cls_logits.shape == (
        1,
        12,
        20,
        20,
    ), f"Unexpected cls shape: {cls_logits.shape}"
    assert reg_preds.shape == (1, 4, 20, 20), f"Unexpected reg shape: {reg_preds.shape}"
    assert centerness.shape == (
        1,
        1,
        20,
        20,
    ), f"Unexpected centerness shape: {centerness.shape}"


def test_detection_head_param_count():
    """Sanity check the head stays lightweight."""
    from src.student.head import DetectionHead

    model = DetectionHead(in_channels=256, num_classes=12)
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Detection head parameter count: {num_params:,}")
    assert num_params < 2_000_000, f"Head too large: {num_params:,} params"
