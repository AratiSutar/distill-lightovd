"""
Training loop for distilling the OWL-ViT teacher's knowledge into the
lightweight student detector.
"""

import os

import torch
from torch.utils.data import DataLoader

from src.data.dataset import PseudoLabelDataset, collate_fn, CLASS_NAMES
from src.distillation.losses import DistillationLoss, compute_class_weights
from src.student.detector import StudentDetector


def train_one_epoch(model, loss_fn, dataloader, optimizer, device):
    """
    Run one full pass over the dataloader, updating model weights.

    Returns:
        dict of average loss values over the epoch.
    """
    model.train()

    total_losses = {
        "total_loss": 0.0,
        "cls_loss": 0.0,
        "reg_loss": 0.0,
        "centerness_loss": 0.0,
    }
    num_batches = 0

    for images, boxes_list, labels_list in dataloader:
        images = images.to(device)
        boxes_list = [b.to(device) for b in boxes_list]
        labels_list = [lbl.to(device) for lbl in labels_list]

        optimizer.zero_grad()

        cls_logits, reg_preds, centerness_preds = model(images)
        result = loss_fn(
            cls_logits, reg_preds, centerness_preds, boxes_list, labels_list
        )

        result["total_loss"].backward()
        optimizer.step()

        for key in total_losses:
            total_losses[key] += result[key].item()
        num_batches += 1

    avg_losses = {key: val / max(num_batches, 1) for key, val in total_losses.items()}
    return avg_losses


def save_checkpoint(model, optimizer, epoch, checkpoint_path):
    """Save model + optimizer state so training can resume later."""
    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        checkpoint_path,
    )


def load_checkpoint(model, optimizer, checkpoint_path, device):
    """Load a saved checkpoint, returning the epoch to resume from."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["epoch"]


def train(
    image_dir,
    labels_path,
    checkpoint_dir,
    num_epochs=30,
    batch_size=8,
    learning_rate=1e-3,
    resume=True,
):
    """
    Full training entrypoint: sets up model, data, optimizer, and runs
    the training loop with checkpointing.
    """
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Training on device: {device}")

    os.makedirs(checkpoint_dir, exist_ok=True)

    dataset = PseudoLabelDataset(image_dir, labels_path, image_size=320)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn,
        num_workers=2,
    )
    print(f"Dataset size: {len(dataset)} images")

    # compute class weights from actual pseudo-label distribution
    class_counts = {i: 0 for i in range(len(CLASS_NAMES))}
    for data in dataset.pseudo_labels.values():
        for label_name in data["labels"]:
            class_counts[CLASS_NAMES.index(label_name)] += 1
    class_weights = compute_class_weights(
        class_counts, num_classes=len(CLASS_NAMES)
    ).to(device)
    print(f"Class weights: {class_weights}")

    model = StudentDetector(num_classes=len(CLASS_NAMES)).to(device)
    loss_fn = DistillationLoss(
        num_classes=len(CLASS_NAMES), class_weights=class_weights
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

    start_epoch = 0
    checkpoint_path = os.path.join(checkpoint_dir, "latest.pt")
    if resume and os.path.exists(checkpoint_path):
        start_epoch = load_checkpoint(model, optimizer, checkpoint_path, device) + 1
        print(f"Resumed from checkpoint, starting at epoch {start_epoch}")

    for epoch in range(start_epoch, num_epochs):
        avg_losses = train_one_epoch(model, loss_fn, dataloader, optimizer, device)
        print(
            f"Epoch {epoch+1}/{num_epochs} - "
            f"total: {avg_losses['total_loss']:.4f}, "
            f"cls: {avg_losses['cls_loss']:.4f}, "
            f"reg: {avg_losses['reg_loss']:.4f}, "
            f"centerness: {avg_losses['centerness_loss']:.4f}"
        )

        save_checkpoint(model, optimizer, epoch, checkpoint_path)

        if (epoch + 1) % 5 == 0:
            epoch_checkpoint_path = os.path.join(checkpoint_dir, f"epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, epoch_checkpoint_path)

    print("Training complete.")
    return model
