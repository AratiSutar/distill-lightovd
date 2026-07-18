"""
Tests for FCOS-style target assignment logic.
"""

import torch


def test_generate_grid_centers_shape():
    """Confirm grid centers have the right shape for a 20x20 feature map."""
    from src.distillation.target_assignment import generate_grid_centers

    centers = generate_grid_centers(feature_h=20, feature_w=20, stride=16)

    assert centers.shape == (400, 2)  # 20*20 = 400 cells, each with (x, y)


def test_generate_grid_centers_values():
    """Confirm the first and last grid cell centers are at expected pixel locations."""
    from src.distillation.target_assignment import generate_grid_centers

    centers = generate_grid_centers(feature_h=20, feature_w=20, stride=16)

    # first cell (top-left): center should be at (0.5*16, 0.5*16) = (8, 8)
    first_center = centers[0]
    assert torch.allclose(first_center, torch.tensor([8.0, 8.0]))

    # last cell (bottom-right, index 19,19): center should be at (19.5*16, 19.5*16) = (312, 312)
    last_center = centers[-1]
    assert torch.allclose(last_center, torch.tensor([312.0, 312.0]))


def test_assign_targets_single_box_center():
    """
    Place one box roughly in the center of a 320x320 image and confirm
    the cell at the box's center gets assigned correctly.
    """
    from src.distillation.target_assignment import (
        generate_grid_centers,
        assign_targets_single_image,
    )

    centers = generate_grid_centers(feature_h=20, feature_w=20, stride=16)

    # a box from (100,100) to (200,200) -> center at (150, 150)
    gt_boxes = torch.tensor([[100.0, 100.0, 200.0, 200.0]])
    gt_labels = torch.tensor([3])  # arbitrary class index, e.g. "motorcycle"

    cls_targets, reg_targets, centerness_targets = assign_targets_single_image(
        centers, gt_boxes, gt_labels, num_classes=12
    )

    print(f"\nNumber of positive cells: {(cls_targets != -1).sum().item()}")
    print(f"Unique assigned classes: {cls_targets[cls_targets != -1].unique()}")

    # the grid cell whose center is closest to (150, 150) should be positive
    # cell center formula: (i + 0.5) * 16 -> i=9 gives (9.5*16)=152, close to 150
    target_cell_idx = 9 * 20 + 9  # row 9, col 9 in the flattened grid

    assert (
        cls_targets[target_cell_idx] == 3
    ), "Center cell should be assigned to our box's class"

    # centerness at the exact center should be close to 1.0 (high confidence)
    center_centerness = centerness_targets[target_cell_idx]
    print(f"Centerness at center cell: {center_centerness.item():.3f}")
    assert center_centerness > 0.8, "Centerness near box center should be high"

    # a cell far outside the box should be background (-1)
    far_cell_idx = 0  # top-left corner, definitely outside the box
    assert cls_targets[far_cell_idx] == -1, "Cell outside the box should be background"


def test_assign_targets_no_boxes():
    """Confirm an image with zero ground-truth boxes returns all-background targets."""
    from src.distillation.target_assignment import (
        generate_grid_centers,
        assign_targets_single_image,
    )

    centers = generate_grid_centers(feature_h=20, feature_w=20, stride=16)
    gt_boxes = torch.zeros((0, 4))
    gt_labels = torch.zeros((0,), dtype=torch.long)

    cls_targets, reg_targets, centerness_targets = assign_targets_single_image(
        centers, gt_boxes, gt_labels, num_classes=12
    )

    assert (
        cls_targets == -1
    ).all(), "All cells should be background when there are no boxes"
