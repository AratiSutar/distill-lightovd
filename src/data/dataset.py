"""
Dataset loader for the distillation training pipeline.
Reads images and OWL-ViT-generated pseudo-labels, prepares them for the
student detector.
"""

import os
import json

from PIL import Image
import torch
from torch.utils.data import Dataset
import torchvision.transforms as T

CLASS_NAMES = [
    "person",
    "car",
    "bicycle",
    "motorcycle",
    "bus",
    "truck",
    "traffic light",
    "stop sign",
    "fire hydrant",
    "dog",
    "backpack",
    "handbag",
]
CLASS_TO_IDX = {name: idx for idx, name in enumerate(CLASS_NAMES)}


class PseudoLabelDataset(Dataset):
    """
    Loads images and their OWL-ViT pseudo-labels for student training.

    Each item returns:
        image: FloatTensor (3, 320, 320), normalized
        boxes: FloatTensor (N, 4) in [x1, y1, x2, y2] format, scaled to 320x320
        labels: LongTensor (N,) class indices
    """

    def __init__(self, image_dir, labels_path, image_size=320):
        self.image_dir = image_dir
        self.image_size = image_size

        with open(labels_path, "r") as f:
            self.pseudo_labels = json.load(f)

        # keep only images that actually have at least one detection
        self.filenames = [
            fname
            for fname, data in self.pseudo_labels.items()
            if len(data["boxes"]) > 0
        ]

        self.transform = T.Compose(
            [
                T.Resize((image_size, image_size)),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        filename = self.filenames[idx]
        filepath = os.path.join(self.image_dir, filename)
        image = Image.open(filepath).convert("RGB")

        orig_w, orig_h = image.size
        scale_x = self.image_size / orig_w
        scale_y = self.image_size / orig_h

        data = self.pseudo_labels[filename]
        boxes = []
        labels = []
        for box, label_name in zip(data["boxes"], data["labels"]):
            x1, y1, x2, y2 = box
            scaled_box = [x1 * scale_x, y1 * scale_y, x2 * scale_x, y2 * scale_y]
            boxes.append(scaled_box)
            labels.append(CLASS_TO_IDX[label_name])

        image_tensor = self.transform(image)
        boxes_tensor = torch.tensor(boxes, dtype=torch.float32)
        labels_tensor = torch.tensor(labels, dtype=torch.long)

        return image_tensor, boxes_tensor, labels_tensor


def collate_fn(batch):
    """
    Custom collate function since each image can have a different number
    of boxes — can't just stack them like normal tensors.
    """
    images = torch.stack([item[0] for item in batch])
    boxes = [item[1] for item in batch]
    labels = [item[2] for item in batch]
    return images, boxes, labels
