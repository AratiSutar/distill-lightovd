"""
Decodes raw student model outputs (classification logits, regression deltas,
centerness) into final detected boxes, scores, and class labels.
"""

import torch
from torchvision.ops import batched_nms


def decode_predictions(
    cls_logits,
    reg_preds,
    centerness_preds,
    stride=16,
    conf_threshold=0.2,
    nms_threshold=0.3,
):
    """
    Convert raw model outputs into final boxes, scores, and class labels.
    Assumes batch size 1.

    Args:
        cls_logits: (1, num_classes, H, W) raw classification logits
        reg_preds: (1, 4, H, W) raw regression predictions
        centerness_preds: (1, 1, H, W) raw centerness predictions
        stride: feature map stride relative to input image
        conf_threshold: minimum combined score to keep a detection
        nms_threshold: IoU threshold for non-max suppression

    Returns:
        boxes: (N, 4) final boxes in [x1, y1, x2, y2] format
        scores: (N,) confidence scores
        classes: (N,) class indices
    """
    device = cls_logits.device
    _, num_classes, h, w = cls_logits.shape

    shifts_x = (torch.arange(w, device=device) + 0.5) * stride
    shifts_y = (torch.arange(h, device=device) + 0.5) * stride
    grid_y, grid_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    centers = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    cls_logits = cls_logits.permute(0, 2, 3, 1).reshape(-1, num_classes)
    reg_preds = reg_preds.permute(0, 2, 3, 1).reshape(-1, 4)
    centerness_preds = centerness_preds.permute(0, 2, 3, 1).reshape(-1)

    cls_scores = torch.sigmoid(cls_logits)
    centerness_scores = torch.sigmoid(centerness_preds)

    combined_scores = cls_scores * centerness_scores.unsqueeze(1)

    max_scores, class_ids = combined_scores.max(dim=1)

    keep_mask = max_scores > conf_threshold
    if keep_mask.sum() == 0:
        return (
            torch.empty((0, 4), device=device),
            torch.empty((0,), device=device),
            torch.empty((0,), dtype=torch.long, device=device),
        )

    filtered_centers = centers[keep_mask]
    filtered_reg = reg_preds[keep_mask]
    filtered_scores = max_scores[keep_mask]
    filtered_classes = class_ids[keep_mask]

    x1 = filtered_centers[:, 0] - filtered_reg[:, 0]
    y1 = filtered_centers[:, 1] - filtered_reg[:, 1]
    x2 = filtered_centers[:, 0] + filtered_reg[:, 2]
    y2 = filtered_centers[:, 1] + filtered_reg[:, 3]
    boxes = torch.stack([x1, y1, x2, y2], dim=1)

    keep_indices = batched_nms(boxes, filtered_scores, filtered_classes, nms_threshold)

    return (
        boxes[keep_indices],
        filtered_scores[keep_indices],
        filtered_classes[keep_indices],
    )
