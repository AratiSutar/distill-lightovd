"""
Tests for distillation loss functions.
"""

import torch


def test_compute_class_weights():
    """Confirm rare classes get higher weights than common ones."""
    from src.distillation.losses import compute_class_weights

    class_counts = {
        0: 8702,
        1: 75,
    }  # class 0 = common (person), class 1 = rare (stop sign)
    weights = compute_class_weights(class_counts, num_classes=2)

    print(f"\nClass weights: {weights}")

    assert (
        weights[1] > weights[0]
    ), "Rare class should have a higher weight than common class"


def test_focal_loss_no_positives():
    """Confirm loss is zero (not NaN) when there are no positive cells."""
    from src.distillation.losses import FocalLoss

    loss_fn = FocalLoss(num_classes=12)
    cls_logits = torch.randn(400, 12)
    cls_targets = torch.full((400,), -1, dtype=torch.long)  # all background

    loss = loss_fn(cls_logits, cls_targets)

    print(f"\nLoss with no positives: {loss.item()}")
    assert loss.item() == 0.0
    assert not torch.isnan(loss)


def test_focal_loss_decreases_with_better_predictions():
    """Confirm loss is lower when logits confidently match the correct class."""
    from src.distillation.losses import FocalLoss

    loss_fn = FocalLoss(num_classes=3)

    cls_targets = torch.tensor([0, 1, -1])  # 2 positives, 1 background

    # "bad" prediction: confidently wrong
    bad_logits = torch.tensor([[-5.0, 5.0, -5.0], [5.0, -5.0, -5.0], [0.0, 0.0, 0.0]])

    # "good" prediction: confidently correct
    good_logits = torch.tensor([[5.0, -5.0, -5.0], [-5.0, 5.0, -5.0], [0.0, 0.0, 0.0]])

    bad_loss = loss_fn(bad_logits, cls_targets)
    good_loss = loss_fn(good_logits, cls_targets)

    print(f"\nBad prediction loss: {bad_loss.item():.4f}")
    print(f"Good prediction loss: {good_loss.item():.4f}")

    assert (
        good_loss.item() < bad_loss.item()
    ), "Correct predictions should have lower loss"


def test_focal_loss_with_class_weights():
    """Confirm class weighting amplifies loss for the rare class, holding the mistake identical."""
    from src.distillation.losses import FocalLoss

    class_weights = torch.tensor([1.0, 10.0, 1.0])  # class 1 weighted 10x higher

    # identical confidently-wrong prediction pattern for both cases
    logits = torch.tensor([[-5.0, -5.0, 5.0]])  # confidently predicts class 2

    loss_fn_weighted = FocalLoss(num_classes=3, class_weights=class_weights)
    loss_fn_unweighted = FocalLoss(num_classes=3, class_weights=None)

    # true class is 0 (low weight) - wrong prediction
    target_class0 = torch.tensor([0])
    loss_class0_weighted = loss_fn_weighted(logits, target_class0)
    loss_class0_unweighted = loss_fn_unweighted(logits, target_class0)

    # true class is 1 (high weight, 10x) - same wrong prediction pattern
    target_class1 = torch.tensor([1])
    loss_class1_weighted = loss_fn_weighted(logits, target_class1)
    loss_class1_unweighted = loss_fn_unweighted(logits, target_class1)

    print(
        f"\nClass 0 (weight=1.0) - unweighted: {loss_class0_unweighted.item():.4f}, "
        f"weighted: {loss_class0_weighted.item():.4f}"
    )
    print(
        f"Class 1 (weight=10.0) - unweighted: {loss_class1_unweighted.item():.4f}, "
        f"weighted: {loss_class1_weighted.item():.4f}"
    )

    # unweighted losses should be roughly equal (same mistake pattern)
    assert abs(loss_class0_unweighted.item() - loss_class1_unweighted.item()) < 0.01

    # weighted loss for class 1 (high weight) should be much larger than class 0 (low weight)
    assert loss_class1_weighted.item() > loss_class0_weighted.item() * 5


def test_iou_loss_perfect_match():
    """Confirm loss is ~0 when predicted and target deltas are identical."""
    from src.distillation.losses import iou_loss

    deltas = torch.tensor([[10.0, 10.0, 10.0, 10.0], [5.0, 5.0, 20.0, 20.0]])

    loss = iou_loss(deltas, deltas)

    print(f"\nIoU loss for perfect match: {loss.item():.6f}")
    assert loss.item() < 1e-5


def test_iou_loss_no_overlap():
    """
    Confirm loss is close to 1 when predicted box doesn't overlap target at all.
    We simulate this using extreme delta mismatches.
    """
    from src.distillation.losses import iou_loss

    # target box: reasonably sized box around the center
    target_deltas = torch.tensor([[10.0, 10.0, 10.0, 10.0]])

    # predicted box: tiny sliver far in one direction (near-zero overlap)
    pred_deltas = torch.tensor([[0.01, 0.01, 0.01, 0.01]])

    loss = iou_loss(pred_deltas, target_deltas)

    print(f"IoU loss for near-zero overlap: {loss.item():.6f}")
    assert loss.item() > 0.9


def test_iou_loss_partial_overlap_between_extremes():
    """partial overlap gives a loss between the perfect-match and no-overlap cases."""
    from src.distillation.losses import iou_loss

    target_deltas = torch.tensor([[10.0, 10.0, 10.0, 10.0]])
    pred_deltas = torch.tensor(
        [[5.0, 5.0, 15.0, 15.0]]
    )  # bigger box, same center, partial overlap

    loss = iou_loss(pred_deltas, target_deltas)

    print(f"IoU loss for partial overlap: {loss.item():.6f}")
    assert 0.0 < loss.item() < 1.0


def test_iou_loss_with_weights():
    """Confirm per-cell weighting changes the aggregated loss appropriately."""
    from src.distillation.losses import iou_loss

    target_deltas = torch.tensor([[10.0, 10.0, 10.0, 10.0], [10.0, 10.0, 10.0, 10.0]])
    pred_deltas = torch.tensor(
        [[5.0, 5.0, 15.0, 15.0], [10.0, 10.0, 10.0, 10.0]]
    )  # cell0: partial overlap, cell1: perfect

    weights = torch.tensor([1.0, 0.0])  # only cell 0 counts
    loss_weighted = iou_loss(pred_deltas, target_deltas, weights=weights)

    loss_unweighted = iou_loss(pred_deltas, target_deltas)

    print(f"Weighted loss (only cell 0 counts): {loss_weighted.item():.6f}")
    print(f"Unweighted loss (both cells average): {loss_unweighted.item():.6f}")

    # weighted loss should be higher since it's fully driven by the imperfect cell 0
    assert loss_weighted.item() > loss_unweighted.item()


def test_centerness_loss_no_positives():
    """Confirm loss is zero (not NaN) when there are no positive cells."""
    from src.distillation.losses import centerness_loss

    pred = torch.randn(400)
    target = torch.zeros(400)
    positive_mask = torch.zeros(400, dtype=torch.bool)  # no positives

    loss = centerness_loss(pred, target, positive_mask)

    print(f"\nCenterness loss with no positives: {loss.item()}")
    assert loss.item() == 0.0
    assert not torch.isnan(loss)


def test_centerness_loss_perfect_prediction():
    """Confirm loss is near-zero when predictions confidently match targets."""
    from src.distillation.losses import centerness_loss

    # target centerness values close to 1.0 and 0.0
    target = torch.tensor([1.0, 0.0, 1.0])
    # logits that, after sigmoid, closely match those targets
    pred = torch.tensor([10.0, -10.0, 10.0])  # sigmoid(10)~1.0, sigmoid(-10)~0.0
    positive_mask = torch.tensor([True, True, True])

    loss = centerness_loss(pred, target, positive_mask)

    print(f"Centerness loss for near-perfect predictions: {loss.item():.6f}")
    assert loss.item() < 0.01


def test_centerness_loss_wrong_prediction():
    """Confirm loss is high when predictions are confidently wrong."""
    from src.distillation.losses import centerness_loss

    target = torch.tensor([1.0, 0.0])
    pred = torch.tensor([-10.0, 10.0])  # exactly backwards
    positive_mask = torch.tensor([True, True])

    loss = centerness_loss(pred, target, positive_mask)

    print(f"Centerness loss for wrong predictions: {loss.item():.4f}")
    assert loss.item() > 5.0


def test_centerness_loss_ignores_negative_cells():
    """Confirm background cells (positive_mask=False) don't affect the loss."""
    from src.distillation.losses import centerness_loss

    target = torch.tensor([1.0, 0.0, 0.5])
    pred_good = torch.tensor(
        [10.0, -10.0, 999.0]
    )  # cell 2 wildly wrong, but masked out
    positive_mask = torch.tensor([True, True, False])
    loss = centerness_loss(pred_good, target, positive_mask)

    print(f"Centerness loss ignoring masked-out bad cell: {loss.item():.6f}")
    assert loss.item() < 0.01  # should stay low since cell 2 is excluded


def test_distillation_loss_end_to_end():
    """
    Full integration test: run dummy model outputs and ground truth through
    the combined DistillationLoss and confirm it produces sensible results.
    """
    from src.distillation.losses import DistillationLoss

    batch_size = 2
    num_classes = 12
    feature_h, feature_w = 20, 20

    loss_fn = DistillationLoss(
        num_classes=num_classes, feature_h=feature_h, feature_w=feature_w
    )

    # dummy model outputs, matching StudentDetector's output shapes
    # requires_grad=True simulates real outputs coming from an actual model
    cls_logits = torch.randn(
        batch_size, num_classes, feature_h, feature_w, requires_grad=True
    )
    reg_preds = (torch.rand(batch_size, 4, feature_h, feature_w) * 20).requires_grad_()
    centerness_preds = torch.randn(
        batch_size, 1, feature_h, feature_w, requires_grad=True
    )

    # dummy ground truth: image 0 has 2 boxes, image 1 has 1 box
    gt_boxes_list = [
        torch.tensor([[50.0, 50.0, 150.0, 150.0], [200.0, 100.0, 280.0, 200.0]]),
        torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
    ]
    gt_labels_list = [
        torch.tensor([0, 3]),
        torch.tensor([5]),
    ]

    result = loss_fn(
        cls_logits, reg_preds, centerness_preds, gt_boxes_list, gt_labels_list
    )

    print(f"\nTotal loss: {result['total_loss'].item():.4f}")
    print(f"Cls loss: {result['cls_loss'].item():.4f}")
    print(f"Reg loss: {result['reg_loss'].item():.4f}")
    print(f"Centerness loss: {result['centerness_loss'].item():.4f}")

    assert not torch.isnan(result["total_loss"])
    assert result["total_loss"].item() > 0
    assert result["total_loss"].requires_grad  # must be differentiable for backprop


def test_distillation_loss_backward_pass_works():
    """Confirm gradients actually flow through the whole loss (critical for training)."""
    from src.distillation.losses import DistillationLoss
    from src.student.detector import StudentDetector

    model = StudentDetector(num_classes=12)
    loss_fn = DistillationLoss(num_classes=12, feature_h=20, feature_w=20)

    dummy_image = torch.randn(1, 3, 320, 320)
    cls_logits, reg_preds, centerness_preds = model(dummy_image)

    gt_boxes_list = [torch.tensor([[50.0, 50.0, 150.0, 150.0]])]
    gt_labels_list = [torch.tensor([2])]

    result = loss_fn(
        cls_logits, reg_preds, centerness_preds, gt_boxes_list, gt_labels_list
    )
    result["total_loss"].backward()

    # confirm at least one model parameter received a gradient
    has_gradient = any(
        p.grad is not None and p.grad.abs().sum() > 0 for p in model.parameters()
    )
    print(f"\nModel received gradients: {has_gradient}")
    assert has_gradient
