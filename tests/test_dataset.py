"""
Tests for the PseudoLabelDataset loader, using synthetic dummy data.
"""

import json
from PIL import Image


def test_pseudo_label_dataset_loads_correctly(tmp_path):
    """Create fake images + labels, confirm the dataset loads and shapes are correct."""
    from src.data.dataset import PseudoLabelDataset

    image_dir = tmp_path / "images"
    image_dir.mkdir()

    img1 = Image.new("RGB", (640, 480), color="red")
    img1.save(image_dir / "img1.jpg")

    img2 = Image.new("RGB", (500, 500), color="blue")
    img2.save(image_dir / "img2.jpg")

    labels_data = {
        "img1.jpg": {
            "boxes": [[10, 10, 100, 100], [200, 150, 300, 250]],
            "scores": [0.8, 0.6],
            "labels": ["car", "person"],
        },
        "img2.jpg": {
            "boxes": [[50, 50, 150, 150]],
            "scores": [0.9],
            "labels": ["dog"],
        },
    }
    labels_path = tmp_path / "pseudo_labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels_data, f)

    dataset = PseudoLabelDataset(str(image_dir), str(labels_path), image_size=320)

    print(f"\nDataset size: {len(dataset)}")

    image_tensor, boxes_tensor, labels_tensor = dataset[0]

    print(f"Image tensor shape: {image_tensor.shape}")
    print(f"Boxes tensor shape: {boxes_tensor.shape}")
    print(f"Boxes tensor values:\n{boxes_tensor}")
    print(f"Labels tensor: {labels_tensor}")

    assert len(dataset) == 2
    assert image_tensor.shape == (3, 320, 320)
    assert boxes_tensor.shape[1] == 4
    assert labels_tensor.dtype.is_floating_point is False
