import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from models.unet import UNet3D
from models.dataset import MRISegmentationDataset


DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


class DiceLoss(nn.Module):
    """Dice loss for binary segmentation."""

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.smooth = smooth

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(predictions)

        predictions = predictions.reshape(
            predictions.size(0), -1
        )

        targets = targets.reshape(
            targets.size(0), -1
        )

        intersection = (
            predictions * targets
        ).sum(dim=1)

        dice = (
            2.0 * intersection + self.smooth
        ) / (
            predictions.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


def dice_score(predictions, targets):

    predictions = torch.sigmoid(predictions)

    predictions = (
        predictions > 0.5
    ).float()

    predictions = predictions.reshape(
        predictions.size(0), -1
    )

    targets = targets.reshape(
        targets.size(0), -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    dice = (
        2.0 * intersection + 1e-6
    ) / (
        predictions.sum(dim=1)
        + targets.sum(dim=1)
        + 1e-6
    )

    return dice.mean().item()


def iou_score(predictions, targets):

    predictions = torch.sigmoid(predictions)

    predictions = (
        predictions > 0.5
    ).float()

    predictions = predictions.reshape(
        predictions.size(0), -1
    )

    targets = targets.reshape(
        targets.size(0), -1
    )

    intersection = (
        predictions * targets
    ).sum(dim=1)

    union = (
        predictions
        + targets
        - predictions * targets
    ).sum(dim=1)

    iou = (
        intersection + 1e-6
    ) / (
        union + 1e-6
    )

    return iou.mean().item()


def load_hospital_data(hospital_id):

    data_path = f"datasets/hospital_{hospital_id}"

    dataset = MRISegmentationDataset(
        data_path
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=True
    )

    return dataset, loader


def train_local_model(
    model,
    train_loader,
    epochs,
    learning_rate
):

    model.train()

    criterion = DiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=learning_rate
    )

    total_loss = 0.0

    for epoch in range(epochs):

        epoch_loss = 0.0

        for images, masks in train_loader:

            images = images.to(DEVICE)

            masks = masks.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        total_loss = (
            epoch_loss / len(train_loader)
        )

    return total_loss


def evaluate_local_model(
    model,
    loader
):

    model.eval()

    criterion = DiceLoss()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    batches = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(DEVICE)

            masks = masks.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                masks
            )

            total_loss += loss.item()

            total_dice += dice_score(
                outputs,
                masks
            )

            total_iou += iou_score(
                outputs,
                masks
            )

            batches += 1

    if batches == 0:
        return 0.0, 0.0, 0.0

    return (
        total_loss / batches,
        total_dice / batches,
        total_iou / batches
    )