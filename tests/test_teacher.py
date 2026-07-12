"""
Tests for the OWL-ViT teacher wrapper.
"""

import pytest
from PIL import Image


def test_owlvit_teacher_importable():
    """Sanity check that the teacher module and class are importable."""
    from src.teacher.inference import OwlViTTeacher  # noqa: F401


@pytest.mark.slow
def test_owlvit_teacher_detect_runs():
    """
    Integration test: loads real OWL-ViT weights and runs detection
    on a blank dummy image. Marked 'slow' since it downloads model weights.
    Skip in fast CI runs with: pytest -m "not slow"
    """
    from src.teacher.inference import OwlViTTeacher

    teacher = OwlViTTeacher(device="cpu")
    dummy_image = Image.new("RGB", (224, 224), color="white")

    result = teacher.detect(dummy_image, ["a photo of a cat"], threshold=0.1)

    assert "boxes" in result
    assert "scores" in result
    assert "labels" in result
    assert isinstance(result["boxes"], list)