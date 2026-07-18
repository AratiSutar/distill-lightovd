"""
FCOS-style target assignment: converts ground-truth boxes into
per-grid-cell training targets for the student detector.
"""

import torch


def generate_grid_centers(feature_h, feature_w, stride):
    """
    Compute the (x, y) center coordinates, in original image pixels,
    for every cell in the feature map grid.

    Returns:
        Tensor of shape (feature_h * feature_w, 2) — (x, y) per grid cell.
    """
    shifts_x = (torch.arange(feature_w) + 0.5) * stride
    shifts_y = (torch.arange(feature_h) + 0.5) * stride

    grid_y, grid_x = torch.meshgrid(shifts_y, shifts_x, indexing="ij")
    centers = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=1)

    return centers  # shape: (H*W, 2)


def assign_targets_single_image(centers, gt_boxes, gt_labels, num_classes):
    """
    Assign ground-truth boxes to grid cells for a single image.

    Args:
        centers: (H*W, 2) grid cell center coordinates (from generate_grid_centers)
        gt_boxes: (N, 4) ground-truth boxes in [x1, y1, x2, y2] format
        gt_labels: (N,) ground-truth class indices
        num_classes: total number of classes

    Returns:
        cls_targets: (H*W,) class index per cell, or -1 for background (no object)
        reg_targets: (H*W, 4) distances (left, top, right, bottom) per cell
        centerness_targets: (H*W,) centerness value per cell
    """
    num_cells = centers.shape[0]
    num_boxes = gt_boxes.shape[0]

    if num_boxes == 0:
        # no objects in this image — everything is background
        cls_targets = torch.full((num_cells,), -1, dtype=torch.long)
        reg_targets = torch.zeros((num_cells, 4))
        centerness_targets = torch.zeros((num_cells,))
        return cls_targets, reg_targets, centerness_targets

    cx = centers[:, 0].unsqueeze(1)  # (H*W, 1)
    cy = centers[:, 1].unsqueeze(1)  # (H*W, 1)

    x1 = gt_boxes[:, 0].unsqueeze(0)  # (1, N)
    y1 = gt_boxes[:, 1].unsqueeze(0)
    x2 = gt_boxes[:, 2].unsqueeze(0)
    y2 = gt_boxes[:, 3].unsqueeze(0)

    # distance from each cell center to each box's edges
    left = cx - x1  # (H*W, N)
    top = cy - y1
    right = x2 - cx
    bottom = y2 - cy

    # a cell is "inside" a box if all 4 distances are positive
    is_inside = (left > 0) & (top > 0) & (right > 0) & (bottom > 0)  # (H*W, N)

    # box areas, used to pick the smallest box when a cell is inside multiple boxes
    areas = (gt_boxes[:, 2] - gt_boxes[:, 0]) * (
        gt_boxes[:, 3] - gt_boxes[:, 1]
    )  # (N,)
    areas = areas.unsqueeze(0).expand(num_cells, num_boxes).clone()  # (H*W, N)
    areas[~is_inside] = float("inf")  # ignore boxes the cell isn't inside

    min_areas, matched_box_idx = areas.min(dim=1)  # (H*W,)

    is_positive = min_areas < float("inf")  # cells that matched at least one box

    cls_targets = torch.full((num_cells,), -1, dtype=torch.long)
    cls_targets[is_positive] = gt_labels[matched_box_idx[is_positive]]

    reg_targets = torch.zeros((num_cells, 4))
    reg_targets[is_positive, 0] = left[is_positive, matched_box_idx[is_positive]]
    reg_targets[is_positive, 1] = top[is_positive, matched_box_idx[is_positive]]
    reg_targets[is_positive, 2] = right[is_positive, matched_box_idx[is_positive]]
    reg_targets[is_positive, 3] = bottom[is_positive, matched_box_idx[is_positive]]

    # centerness: sqrt( (min(l,r)/max(l,r)) * (min(t,b)/max(t,b)) )
    centerness_targets = torch.zeros((num_cells,))
    l, t, r, b = (
        reg_targets[:, 0],
        reg_targets[:, 1],
        reg_targets[:, 2],
        reg_targets[:, 3],
    )
    lr_ratio = torch.min(l, r) / (torch.max(l, r) + 1e-6)
    tb_ratio = torch.min(t, b) / (torch.max(t, b) + 1e-6)
    centerness_targets[is_positive] = torch.sqrt((lr_ratio * tb_ratio).clamp(min=0))[
        is_positive
    ]

    return cls_targets, reg_targets, centerness_targets
