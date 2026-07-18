"""
Tests for the training loop.
"""

import torch
from torch.utils.data import DataLoader


def test_train_one_epoch_runs_and_reduces_loss():
    """
    Confirm train_one_epoch runs without error, and that loss decreases
    over a few epochs on a tiny fixed dummy batch (sanity check that
    the model can actually learn something).
    """
    from src.distillation.train import train_one_epoch
    from src.distillation.losses import DistillationLoss
    from src.student.detector import StudentDetector

    device = "cpu"
    model = StudentDetector(num_classes=12).to(device)
    loss_fn = DistillationLoss(num_classes=12, feature_h=20, feature_w=20)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    # tiny fixed dummy dataset: 2 images, fixed boxes/labels
    images = torch.randn(2, 3, 320, 320)
    boxes_list = [
        torch.tensor([[50.0, 50.0, 150.0, 150.0]]),
        torch.tensor([[100.0, 100.0, 200.0, 200.0]]),
    ]
    labels_list = [torch.tensor([2]), torch.tensor([5])]

    class DummyDataset(torch.utils.data.Dataset):
        def __len__(self):
            return 2

        def __getitem__(self, idx):
            return images[idx], boxes_list[idx], labels_list[idx]

    def dummy_collate(batch):
        imgs = torch.stack([b[0] for b in batch])
        boxes = [b[1] for b in batch]
        labels = [b[2] for b in batch]
        return imgs, boxes, labels

    dataloader = DataLoader(DummyDataset(), batch_size=2, collate_fn=dummy_collate)

    losses_over_time = []
    for _ in range(5):
        avg_losses = train_one_epoch(model, loss_fn, dataloader, optimizer, device)
        losses_over_time.append(avg_losses["total_loss"])

    print(f"\nLoss over 5 epochs on fixed dummy batch: {losses_over_time}")

    assert all(not torch.isnan(torch.tensor(loss)) for loss in losses_over_time)
    # loss should generally trend downward when overfitting to a tiny fixed batch
    assert losses_over_time[-1] < losses_over_time[0]


def test_checkpoint_save_and_load(tmp_path):
    """Confirm a saved checkpoint can be loaded back and restores model/optimizer state."""
    from src.distillation.train import save_checkpoint, load_checkpoint
    from src.student.detector import StudentDetector

    device = "cpu"
    model = StudentDetector(num_classes=12).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

    checkpoint_path = tmp_path / "test_checkpoint.pt"
    save_checkpoint(model, optimizer, epoch=3, checkpoint_path=str(checkpoint_path))

    assert checkpoint_path.exists()

    # create a fresh model/optimizer and load the checkpoint into them
    new_model = StudentDetector(num_classes=12).to(device)
    new_optimizer = torch.optim.AdamW(new_model.parameters(), lr=1e-3)

    loaded_epoch = load_checkpoint(
        new_model, new_optimizer, str(checkpoint_path), device
    )

    print(f"\nLoaded checkpoint from epoch: {loaded_epoch}")
    assert loaded_epoch == 3

    # confirm the loaded model's weights match the original model's weights
    for p1, p2 in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p1, p2)
