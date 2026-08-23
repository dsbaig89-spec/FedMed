import os
import json
import torch
import torch.nn as nn

from torch.utils.data import DataLoader, random_split

from models.unet import UNet3D
from models.dataset import MRISegmentationDataset


# ==========================================
# Configuration
# ==========================================

DATA_DIR = "datasets/hospital_a"

CHECKPOINT_DIR = "models/checkpoints"

HISTORY_FILE = "baseline/training_history.json"

EPOCHS = 5

BATCH_SIZE = 1

LEARNING_RATE = 0.001

VALIDATION_SPLIT = 0.2

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ==========================================
# Dice Loss
# ==========================================

class DiceLoss(nn.Module):

    def __init__(self, smooth=1e-6):

        super().__init__()

        self.smooth = smooth

    def forward(self, predictions, targets):

        predictions = torch.sigmoid(predictions)

        predictions = predictions.view(
            predictions.size(0),
            -1
        )

        targets = targets.view(
            targets.size(0),
            -1
        )

        intersection = (
            predictions * targets
        ).sum(dim=1)

        dice = (
            2.0 * intersection
            + self.smooth
        ) / (
            predictions.sum(dim=1)
            + targets.sum(dim=1)
            + self.smooth
        )

        return 1.0 - dice.mean()


# ==========================================
# Dice Score
# ==========================================

def dice_score(
    predictions,
    targets,
    threshold=0.5
):

    predictions = torch.sigmoid(
        predictions
    )

    predictions = (
        predictions > threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0),
        -1
    )

    targets = targets.view(
        targets.size(0),
        -1
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


# ==========================================
# IoU
# ==========================================

def iou_score(
    predictions,
    targets,
    threshold=0.5
):

    predictions = torch.sigmoid(
        predictions
    )

    predictions = (
        predictions > threshold
    ).float()

    predictions = predictions.view(
        predictions.size(0),
        -1
    )

    targets = targets.view(
        targets.size(0),
        -1
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


# ==========================================
# Validation
# ==========================================

def validate(
    model,
    loader,
    criterion,
    device
):

    model.eval()

    total_loss = 0.0
    total_dice = 0.0
    total_iou = 0.0

    batches = 0

    with torch.no_grad():

        for images, masks in loader:

            images = images.to(device)

            masks = masks.to(device)

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


# ==========================================
# Main Training
# ==========================================

def main():

    print("=" * 60)
    print("FedMed Centralized Training")
    print("=" * 60)

    print(
        f"\nDevice: {DEVICE}"
    )

    # --------------------------------------
    # Create checkpoint directory
    # --------------------------------------

    os.makedirs(
        CHECKPOINT_DIR,
        exist_ok=True
    )

    # --------------------------------------
    # Load dataset
    # --------------------------------------

    print(
        "\nLoading dataset..."
    )

    dataset = MRISegmentationDataset(
        DATA_DIR
    )

    print(
        f"Total samples: {len(dataset)}"
    )

    # --------------------------------------
    # Train / validation split
    # --------------------------------------

    validation_size = int(
        len(dataset) * VALIDATION_SPLIT
    )

    training_size = (
        len(dataset) - validation_size
    )

    train_dataset, validation_dataset = random_split(
        dataset,
        [training_size, validation_size],
        generator=torch.Generator().manual_seed(42)
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(validation_dataset)}"
    )

    # --------------------------------------
    # DataLoaders
    # --------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False
    )

    # --------------------------------------
    # Model
    # --------------------------------------

    model = UNet3D()

    model = model.to(DEVICE)

    # --------------------------------------
    # Loss and optimizer
    # --------------------------------------

    criterion = DiceLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------
    # Training history
    # --------------------------------------

    history = {
        "train_loss": [],
        "validation_loss": [],
        "validation_dice": [],
        "validation_iou": []
    }

    best_dice = -1.0

    # --------------------------------------
    # Training loop
    # --------------------------------------

    for epoch in range(EPOCHS):

        model.train()

        running_loss = 0.0

        for batch_index, (
            images,
            masks
        ) in enumerate(train_loader):

            images = images.to(DEVICE)

            masks = masks.to(DEVICE)

            # Clear gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = model(images)

            # Calculate loss
            loss = criterion(
                outputs,
                masks
            )

            # Backpropagation
            loss.backward()

            # Update weights
            optimizer.step()

            running_loss += loss.item()

        train_loss = (
            running_loss / len(train_loader)
        )

        # ----------------------------------
        # Validation
        # ----------------------------------

        val_loss, val_dice, val_iou = validate(
            model,
            validation_loader,
            criterion,
            DEVICE
        )

        # Store history
        history["train_loss"].append(
            train_loss
        )

        history["validation_loss"].append(
            val_loss
        )

        history["validation_dice"].append(
            val_dice
        )

        history["validation_iou"].append(
            val_iou
        )

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        print(
            f"Train Loss      : {train_loss:.4f}"
        )

        print(
            f"Validation Loss  : {val_loss:.4f}"
        )

        print(
            f"Validation Dice  : {val_dice:.4f}"
        )

        print(
            f"Validation IoU   : {val_iou:.4f}"
        )

        # ----------------------------------
        # Save best model
        # ----------------------------------

        if val_dice > best_dice:

            best_dice = val_dice

            checkpoint_path = os.path.join(
                CHECKPOINT_DIR,
                "centralized_best.pt"
            )

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "epoch":
                        epoch + 1,

                    "dice":
                        val_dice,

                    "iou":
                        val_iou
                },
                checkpoint_path
            )

            print(
                "✓ Best model checkpoint saved."
            )

    # --------------------------------------
    # Save training history
    # --------------------------------------

    with open(
        HISTORY_FILE,
        "w"
    ) as file:

        json.dump(
            history,
            file,
            indent=4
        )

    print("\n" + "=" * 60)

    print(
        "Centralized training completed!"
    )

    print(
        f"Best Dice Score: {best_dice:.4f}"
    )

    print(
        f"Checkpoint: {CHECKPOINT_DIR}"
    )

    print(
        f"History: {HISTORY_FILE}"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()