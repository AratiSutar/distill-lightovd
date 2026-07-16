"""
Tests for the full StudentDetector (backbone + head combined).
"""

import torch


def test_student_detector_output_shapes():
    """Confirm the full model produces correctly shaped outputs end-to-end."""
    from src.student.detector import StudentDetector

    model = StudentDetector(num_classes=12)
    dummy_image = torch.randn(1, 3, 320, 320)

    cls_logits, reg_preds, centerness = model(dummy_image)

    assert cls_logits.shape == (1, 12, 20, 20)
    assert reg_preds.shape == (1, 4, 20, 20)
    assert centerness.shape == (1, 1, 20, 20)


def test_student_detector_total_params():
    """Confirm the full model stays within our lightweight target (~3-8M params)."""
    from src.student.detector import StudentDetector

    model = StudentDetector(num_classes=12)
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Total student model parameters: {num_params:,}")
    assert num_params < 8_000_000, f"Model too large: {num_params:,} params"
