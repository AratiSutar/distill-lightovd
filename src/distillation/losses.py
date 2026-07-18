"""
Loss functions for distillation training: classification (focal loss),
box regression (IoU loss), and centerness (BCE).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from src.distillation.target_assignment import (
    generate_grid_centers,
    assign_targets_single_image,
)


def compute_class_weights(class_counts, num_classes):
    """
    Compute inverse-frequency class weights to counter dataset imbalance.

    Args:
        class_counts: dict mapping class_idx -> count (from your dataset)
        num_classes: total number of classes

    Returns:
        Tensor of shape (num_classes,) with normalized weights.
    """
    counts = torch.tensor(
        [class_counts.get(i, 1) for i in range(num_classes)], dtype=torch.float32
    )
    weights = 1.0 / counts
    weights = weights / weights.sum() * num_classes  # normalize so weights average to 1
    return weights


class FocalLoss(nn.Module):
    """
    Focal Loss for classification, with optional per-class weighting.
    Operates on raw logits (num_classes,) per grid cell.
    """

    def __init__(self, num_classes, class_weights=None, gamma=2.0, alpha=0.25):
        super().__init__()
        self.num_classes = num_classes
        self.gamma = gamma
        self.alpha = alpha
        self.class_weights = class_weights  # Tensor (num_classes,) or None

    def forward(self, cls_logits, cls_targets):
        """
        Args:
            cls_logits: (N, num_classes) raw logits, N = number of grid cells
            cls_targets: (N,) class indices, -1 for background

        Returns:
            scalar loss
        """
        positive_mask = cls_targets != -1

        if positive_mask.sum() == 0:
            return cls_logits.sum() * 0.0  # no positives, no loss (avoids NaN)

        # one-hot encode targets for positive cells only
        pos_logits = cls_logits[positive_mask]  # (P, num_classes)
        pos_targets = cls_targets[positive_mask]  # (P,)

        targets_one_hot = F.one_hot(pos_targets, num_classes=self.num_classes).float()

        probs = torch.sigmoid(pos_logits)
        ce_loss = F.binary_cross_entropy_with_logits(
            pos_logits, targets_one_hot, reduction="none"
        )

        p_t = probs * targets_one_hot + (1 - probs) * (1 - targets_one_hot)
        focal_weight = (1 - p_t) ** self.gamma

        alpha_weight = self.alpha * targets_one_hot + (1 - self.alpha) * (
            1 - targets_one_hot
        )

        loss = alpha_weight * focal_weight * ce_loss  # (P, num_classes)

        if self.class_weights is not None:
            class_weight_per_target = self.class_weights[pos_targets].unsqueeze(
                1
            )  # (P, 1)
            loss = loss * class_weight_per_target

        return loss.sum() / positive_mask.sum().clamp(min=1)


def iou_loss(pred_deltas, target_deltas, weights=None):
    """
    IoU-based regression loss. Both pred and target are (left, top, right, bottom)
    distances from a shared center point, so reconstructing box overlap doesn't
    need the actual center coordinates.

    Args:
        pred_deltas: (P, 4) predicted (left, top, right, bottom)
        target_deltas: (P, 4) target (left, top, right, bottom)
        weights: optional (P,) per-cell weights (e.g. centerness), or None

    Returns:
        scalar loss
    """
    pred_left, pred_top, pred_right, pred_bottom = pred_deltas.unbind(dim=1)
    target_left, target_top, target_right, target_bottom = target_deltas.unbind(dim=1)

    pred_area = (pred_left + pred_right) * (pred_top + pred_bottom)
    target_area = (target_left + target_right) * (target_top + target_bottom)

    inter_w = torch.min(pred_left, target_left) + torch.min(pred_right, target_right)
    inter_h = torch.min(pred_top, target_top) + torch.min(pred_bottom, target_bottom)
    inter_area = inter_w.clamp(min=0) * inter_h.clamp(min=0)

    union_area = pred_area + target_area - inter_area
    iou = inter_area / union_area.clamp(min=1e-6)

    loss = 1.0 - iou

    if weights is not None:
        loss = loss * weights
        return loss.sum() / weights.sum().clamp(min=1e-6)

    return loss.mean()


def centerness_loss(pred_centerness, target_centerness, positive_mask):
    """
    Binary cross-entropy loss between predicted and target centerness,
    computed only over positive (foreground) cells.

    Args:
        pred_centerness: (N,) raw centerness logits
        target_centerness: (N,) target centerness values in [0, 1]
        positive_mask: (N,) boolean mask, True for foreground cells

    Returns:
        scalar loss
    """
    if positive_mask.sum() == 0:
        return pred_centerness.sum() * 0.0

    pred_pos = pred_centerness[positive_mask]
    target_pos = target_centerness[positive_mask]

    loss = F.binary_cross_entropy_with_logits(pred_pos, target_pos, reduction="mean")
    return loss


class DistillationLoss(nn.Module):
    """
    Combines classification (focal), regression (IoU), and centerness losses
    into one total loss for training the student detector.
    """

    def __init__(
        self,
        num_classes,
        feature_h=20,
        feature_w=20,
        stride=16,
        class_weights=None,
        cls_loss_weight=1.0,
        reg_loss_weight=1.0,
        centerness_loss_weight=1.0,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.stride = stride
        self.centers = generate_grid_centers(feature_h, feature_w, stride)

        self.focal_loss = FocalLoss(
            num_classes=num_classes, class_weights=class_weights
        )
        self.cls_loss_weight = cls_loss_weight
        self.reg_loss_weight = reg_loss_weight
        self.centerness_loss_weight = centerness_loss_weight

    def forward(
        self, cls_logits, reg_preds, centerness_preds, gt_boxes_list, gt_labels_list
    ):
        """
        Args:
            cls_logits: (B, num_classes, H, W) raw classification logits from the model
            reg_preds: (B, 4, H, W) raw regression predictions from the model
            centerness_preds: (B, 1, H, W) raw centerness predictions from the model
            gt_boxes_list: list of B tensors, each (N_i, 4) ground-truth boxes
            gt_labels_list: list of B tensors, each (N_i,) ground-truth class indices

        Returns:
            dict with total_loss, cls_loss, reg_loss, centerness_loss (all scalars)
        """
        batch_size = cls_logits.shape[0]
        device = cls_logits.device
        centers = self.centers.to(device)

        # reshape predictions from (B, C, H, W) -> (B, H*W, C)
        cls_logits_flat = cls_logits.permute(0, 2, 3, 1).reshape(
            batch_size, -1, self.num_classes
        )
        reg_preds_flat = reg_preds.permute(0, 2, 3, 1).reshape(batch_size, -1, 4)
        centerness_flat = centerness_preds.permute(0, 2, 3, 1).reshape(batch_size, -1)

        all_cls_targets = []
        all_reg_targets = []
        all_centerness_targets = []

        for i in range(batch_size):
            cls_t, reg_t, center_t = assign_targets_single_image(
                centers,
                gt_boxes_list[i].to(device),
                gt_labels_list[i].to(device),
                self.num_classes,
            )
            all_cls_targets.append(cls_t)
            all_reg_targets.append(reg_t)
            all_centerness_targets.append(center_t)

        cls_targets = torch.cat(all_cls_targets)  # (B*H*W,)
        reg_targets = torch.cat(all_reg_targets)  # (B*H*W, 4)
        centerness_targets = torch.cat(all_centerness_targets)  # (B*H*W,)

        cls_logits_flat = cls_logits_flat.reshape(-1, self.num_classes)
        reg_preds_flat = reg_preds_flat.reshape(-1, 4)
        centerness_flat = centerness_flat.reshape(-1)

        positive_mask = cls_targets != -1

        cls_loss = self.focal_loss(cls_logits_flat, cls_targets)

        if positive_mask.sum() > 0:
            reg_loss = iou_loss(
                reg_preds_flat[positive_mask],
                reg_targets[positive_mask],
                weights=centerness_targets[positive_mask],
            )
        else:
            reg_loss = reg_preds_flat.sum() * 0.0

        center_loss = centerness_loss(
            centerness_flat, centerness_targets, positive_mask
        )

        total_loss = (
            self.cls_loss_weight * cls_loss
            + self.reg_loss_weight * reg_loss
            + self.centerness_loss_weight * center_loss
        )

        return {
            "total_loss": total_loss,
            "cls_loss": cls_loss.detach(),
            "reg_loss": reg_loss.detach(),
            "centerness_loss": center_loss.detach(),
        }
