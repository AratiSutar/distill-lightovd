"""
Tests for the student backbone architecture.
"""

import torch


def test_backbone_output_shape():
    """Confirm the backbone produces the expected output shape for a 320x320 input."""
    from src.student.backbone import LightBackbone

    model = LightBackbone()
    dummy_input = torch.randn(1, 3, 320, 320)  # batch=1, RGB, 320x320

    output = model(dummy_input)

    assert output.shape == (1, 256, 20, 20), f"Unexpected shape: {output.shape}"


def test_backbone_param_count():
    """Sanity check that the backbone is genuinely lightweight (a few million params)."""
    from src.student.backbone import LightBackbone

    model = LightBackbone()
    num_params = sum(p.numel() for p in model.parameters())

    print(f"Backbone parameter count: {num_params:,}")
    assert num_params < 5_000_000, f"Backbone too large: {num_params:,} params"
