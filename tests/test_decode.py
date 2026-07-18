"""
Tests for decoding raw model outputs into final detections.
"""

import torch


def test_decode_predictions_no_detections_above_threshold():
    """Confirm empty output when nothing exceeds the confidence threshold."""
    from src.evaluation.decode import decode_predictions

    h, w = 20, 20
    num_classes = 12

    # very negative logits -> near-zero sigmoid scores everywhere
    cls_logits = torch.full((1, num_classes, h, w), -10.0)
    reg_preds = torch.rand(1, 4, h, w) * 20
    centerness_preds = torch.full((1, 1, h, w), -10.0)

    boxes, scores, classes = decode_predictions(
        cls_logits, reg_preds, centerness_preds, conf_threshold=0.2
    )

    print(f"\nDetections with all-low confidence: {len(boxes)}")
    assert len(boxes) == 0


def test_decode_predictions_single_confident_detection():
    """Confirm a single strongly-confident cell produces exactly one detection."""
    from src.evaluation.decode import decode_predictions

    h, w = 20, 20
    num_classes = 12

    cls_logits = torch.full((1, num_classes, h, w), -10.0)
    centerness_preds = torch.full((1, 1, h, w), -10.0)

    # make one specific cell confidently predict class 3
    cls_logits[0, 3, 10, 10] = 10.0
    centerness_preds[0, 0, 10, 10] = 10.0

    reg_preds = torch.full((1, 4, h, w), 20.0)  # reasonable box size everywhere

    boxes, scores, classes = decode_predictions(
        cls_logits, reg_preds, centerness_preds, conf_threshold=0.2
    )

    print(f"Detections with one confident cell: {len(boxes)}")
    print(f"Predicted class: {classes.tolist()}")

    assert len(boxes) == 1
    assert classes[0].item() == 3


def test_decode_predictions_nms_removes_duplicates():
    """Confirm nearby overlapping confident predictions get merged by NMS."""
    from src.evaluation.decode import decode_predictions

    h, w = 20, 20
    num_classes = 12

    cls_logits = torch.full((1, num_classes, h, w), -10.0)
    centerness_preds = torch.full((1, 1, h, w), -10.0)

    # two ADJACENT cells both confidently predict the same class -> overlapping boxes
    cls_logits[0, 5, 10, 10] = 10.0
    cls_logits[0, 5, 10, 11] = 10.0
    centerness_preds[0, 0, 10, 10] = 10.0
    centerness_preds[0, 0, 10, 11] = 10.0

    reg_preds = torch.full(
        (1, 4, h, w), 30.0
    )  # large boxes -> guaranteed overlap between adjacent cells

    boxes, scores, classes = decode_predictions(
        cls_logits, reg_preds, centerness_preds, conf_threshold=0.2, nms_threshold=0.3
    )

    print(f"Detections after NMS on overlapping duplicates: {len(boxes)}")
    assert len(boxes) == 1  # NMS should merge the two into one
