"""
Mean Average Precision (mAP) computation for evaluating the student detector.
"""

import torch


def compute_iou_matrix(boxes1, boxes2):
    """
    Compute IoU between every pair of boxes in boxes1 and boxes2.

    Args:
        boxes1: (N, 4) boxes in [x1, y1, x2, y2] format
        boxes2: (M, 4) boxes in [x1, y1, x2, y2] format

    Returns:
        (N, M) IoU matrix
    """
    area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
    area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # (N, M, 2)
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # (N, M, 2)

    wh = (rb - lt).clamp(min=0)
    inter = wh[:, :, 0] * wh[:, :, 1]  # (N, M)

    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union.clamp(min=1e-6)

    return iou


def compute_average_precision(pred_boxes, pred_scores, gt_boxes, iou_threshold=0.5):
    """
    Compute Average Precision for a single class across one or more images.

    Args:
        pred_boxes: (N, 4) predicted boxes for this class
        pred_scores: (N,) confidence scores, same order as pred_boxes
        gt_boxes: (M, 4) ground-truth boxes for this class
        iou_threshold: IoU required to count a prediction as a true positive

    Returns:
        float AP score
    """
    if len(pred_boxes) == 0:
        return (
            0.0 if len(gt_boxes) > 0 else 1.0
        )  # no predictions: 0 if there were objects to find, else trivially perfect

    if len(gt_boxes) == 0:
        return 0.0  # predictions exist but nothing to match -> all false positives

    # sort predictions by confidence, descending
    order = torch.argsort(pred_scores, descending=True)
    pred_boxes = pred_boxes[order]

    matched_gt = torch.zeros(len(gt_boxes), dtype=torch.bool)
    tp = torch.zeros(len(pred_boxes))
    fp = torch.zeros(len(pred_boxes))

    iou_matrix = compute_iou_matrix(pred_boxes, gt_boxes)  # (N, M)

    for i in range(len(pred_boxes)):
        ious = iou_matrix[i]
        best_iou, best_gt_idx = ious.max(dim=0)

        if best_iou >= iou_threshold and not matched_gt[best_gt_idx]:
            tp[i] = 1
            matched_gt[best_gt_idx] = True
        else:
            fp[i] = 1

    tp_cumsum = torch.cumsum(tp, dim=0)
    fp_cumsum = torch.cumsum(fp, dim=0)

    recalls = tp_cumsum / len(gt_boxes)
    precisions = tp_cumsum / (tp_cumsum + fp_cumsum).clamp(min=1e-6)

    # 11-point interpolation (standard, simple, matches original PASCAL VOC method)
    ap = 0.0
    for t in torch.linspace(0, 1, 11):
        mask = recalls >= t
        p = precisions[mask].max().item() if mask.any() else 0.0
        ap += p / 11.0

    return ap


def compute_map(
    model,
    dataset,
    class_names,
    device,
    conf_threshold=0.2,
    nms_threshold=0.3,
    iou_threshold=0.5,
):
    """
    Compute mean Average Precision across all classes on a given dataset.

    Args:
        model: trained StudentDetector
        dataset: PseudoLabelDataset instance
        class_names: list of class name strings
        device: "cuda" or "cpu"
        conf_threshold, nms_threshold: detection decoding parameters
        iou_threshold: IoU threshold for counting a true positive

    Returns:
        dict with per-class AP and overall mAP
    """
    from src.evaluation.decode import decode_predictions  # see note below

    model.eval()
    num_classes = len(class_names)

    all_pred_boxes = {i: [] for i in range(num_classes)}
    all_pred_scores = {i: [] for i in range(num_classes)}
    all_gt_boxes = {i: [] for i in range(num_classes)}

    with torch.no_grad():
        for idx in range(len(dataset)):
            image, gt_boxes, gt_labels = dataset[idx]
            image = image.unsqueeze(0).to(device)

            cls_logits, reg_preds, centerness_preds = model(image)
            pred_boxes, pred_scores, pred_classes = decode_predictions(
                cls_logits,
                reg_preds,
                centerness_preds,
                conf_threshold=conf_threshold,
                nms_threshold=nms_threshold,
            )

            for cls_id in range(num_classes):
                cls_mask_pred = pred_classes == cls_id
                if cls_mask_pred.sum() > 0:
                    all_pred_boxes[cls_id].append(pred_boxes[cls_mask_pred].cpu())
                    all_pred_scores[cls_id].append(pred_scores[cls_mask_pred].cpu())

                cls_mask_gt = gt_labels == cls_id
                if cls_mask_gt.sum() > 0:
                    all_gt_boxes[cls_id].append(gt_boxes[cls_mask_gt])

    per_class_ap = {}
    valid_aps = []
    for cls_id, cls_name in enumerate(class_names):
        preds_b = (
            torch.cat(all_pred_boxes[cls_id])
            if all_pred_boxes[cls_id]
            else torch.empty((0, 4))
        )
        preds_s = (
            torch.cat(all_pred_scores[cls_id])
            if all_pred_scores[cls_id]
            else torch.empty((0,))
        )
        gts_b = (
            torch.cat(all_gt_boxes[cls_id])
            if all_gt_boxes[cls_id]
            else torch.empty((0, 4))
        )

        if len(gts_b) == 0:
            per_class_ap[cls_name] = (
                None  # no ground truth for this class in the eval set
            )
            continue

        ap = compute_average_precision(
            preds_b, preds_s, gts_b, iou_threshold=iou_threshold
        )
        per_class_ap[cls_name] = ap
        valid_aps.append(ap)

    mean_ap = sum(valid_aps) / len(valid_aps) if valid_aps else 0.0

    return {"per_class_ap": per_class_ap, "mAP": mean_ap}
