"""
Tests for mAP computation.
"""

import torch


def test_iou_matrix_perfect_overlap():
    """Confirm IoU is 1.0 for identical boxes."""
    from src.evaluation.metrics import compute_iou_matrix

    boxes1 = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    boxes2 = torch.tensor([[10.0, 10.0, 50.0, 50.0]])

    iou = compute_iou_matrix(boxes1, boxes2)

    print(f"\nIoU for identical boxes: {iou.item():.4f}")
    assert torch.allclose(iou, torch.tensor([[1.0]]), atol=1e-5)


def test_iou_matrix_no_overlap():
    """Confirm IoU is 0.0 for non-overlapping boxes."""
    from src.evaluation.metrics import compute_iou_matrix

    boxes1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    boxes2 = torch.tensor([[100.0, 100.0, 110.0, 110.0]])

    iou = compute_iou_matrix(boxes1, boxes2)

    print(f"IoU for non-overlapping boxes: {iou.item():.4f}")
    assert iou.item() == 0.0


def test_average_precision_perfect_predictions():
    """
    Confirm AP is 1.0 when every prediction perfectly matches a ground-truth box,
    with high confidence.
    """
    from src.evaluation.metrics import compute_average_precision

    gt_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [100.0, 100.0, 150.0, 150.0]])
    pred_boxes = gt_boxes.clone()  # exact match
    pred_scores = torch.tensor([0.9, 0.95])

    ap = compute_average_precision(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)

    print(f"\nAP for perfect predictions: {ap:.4f}")
    assert ap > 0.99


def test_average_precision_no_predictions_but_objects_exist():
    """Confirm AP is 0.0 when there are ground-truth objects but no predictions."""
    from src.evaluation.metrics import compute_average_precision

    gt_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    pred_boxes = torch.empty((0, 4))
    pred_scores = torch.empty((0,))

    ap = compute_average_precision(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)

    print(f"AP with no predictions: {ap:.4f}")
    assert ap == 0.0


def test_average_precision_completely_wrong_predictions():
    """Confirm AP is low when predictions don't overlap any ground-truth box."""
    from src.evaluation.metrics import compute_average_precision

    gt_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    pred_boxes = torch.tensor([[200.0, 200.0, 250.0, 250.0]])  # nowhere near gt
    pred_scores = torch.tensor([0.9])

    ap = compute_average_precision(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)

    print(f"AP for completely wrong predictions: {ap:.4f}")
    assert ap < 0.1


def test_average_precision_duplicate_detections_penalized():
    """
    Confirm that multiple predictions on the same object don't inflate AP -
    only the highest-confidence one should count as a true positive.
    """
    from src.evaluation.metrics import compute_average_precision

    gt_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0]])
    # two predictions on the SAME object, one confident and correct, one lower-confidence duplicate
    pred_boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [12.0, 12.0, 52.0, 52.0]])
    pred_scores = torch.tensor([0.9, 0.7])

    ap = compute_average_precision(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5)

    print(f"AP with duplicate detections: {ap:.4f}")
    # should still be reasonably high since the top prediction is correct,
    # but the duplicate becomes a false positive
    assert ap > 0.4


def test_compute_map_with_dummy_dataset(tmp_path):
    """
    End-to-end test: create a tiny dummy dataset and an untrained model,
    confirm compute_map runs without error and returns sensible structure.
    """
    import json
    from PIL import Image
    from src.evaluation.metrics import compute_map
    from src.data.dataset import PseudoLabelDataset
    from src.student.detector import StudentDetector

    image_dir = tmp_path / "images"
    image_dir.mkdir()

    img1 = Image.new("RGB", (320, 320), color="red")
    img1.save(image_dir / "img1.jpg")

    labels_data = {
        "img1.jpg": {
            "boxes": [[50.0, 50.0, 150.0, 150.0]],
            "scores": [0.9],
            "labels": ["car"],
        },
    }
    labels_path = tmp_path / "pseudo_labels.json"
    with open(labels_path, "w") as f:
        json.dump(labels_data, f)

    class_names = [
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

    dataset = PseudoLabelDataset(str(image_dir), str(labels_path), image_size=320)
    model = StudentDetector(num_classes=12)
    device = "cpu"

    result = compute_map(model, dataset, class_names, device)

    print(f"\nmAP: {result['mAP']:.4f}")
    print(f"Per-class AP: {result['per_class_ap']}")

    assert "mAP" in result
    assert "per_class_ap" in result
    assert len(result["per_class_ap"]) == 12
    assert 0.0 <= result["mAP"] <= 1.0
    assert result["mAP"] == result["per_class_ap"]["car"]
