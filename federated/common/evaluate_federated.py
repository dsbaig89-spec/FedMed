import json
import os

import torch
from torch.utils.data import DataLoader

from models.unet import UNet3D
from models.dataset import MRISegmentationDataset

from federated.common.federated_utils import (
    DiceLoss,
    dice_score,
    iou_score,
    DEVICE,
)


GLOBAL_MODEL = "models/checkpoints/federated_global.pt"

HOSPITALS = {
    "A": "datasets/hospital_a",
    "B": "datasets/hospital_b",
    "C": "datasets/hospital_c",
}


def evaluate_hospital(model, hospital_path):

    dataset = MRISegmentationDataset(
        hospital_path
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False
    )

    criterion = DiceLoss()

    model.eval()

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


def main():

    print("=" * 60)
    print("FedMed Federated Model Evaluation")
    print("=" * 60)

    if not os.path.exists(GLOBAL_MODEL):

        print(
            "\nFederated global model not found:"
        )

        print(GLOBAL_MODEL)

        print(
            "\nRun federated training first."
        )

        return

    model = UNet3D().to(DEVICE)

    checkpoint = torch.load(
        GLOBAL_MODEL,
        map_location=DEVICE
    )

    if "model_state_dict" in checkpoint:

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

    else:

        model.load_state_dict(
            checkpoint
        )

    results = {}

    for hospital, path in HOSPITALS.items():

        loss, dice, iou = evaluate_hospital(
            model,
            path
        )

        results[hospital] = {
            "loss": loss,
            "dice": dice,
            "iou": iou
        }

        print(
            f"\nHospital {hospital}"
        )

        print(
            f"Loss: {loss:.4f}"
        )

        print(
            f"Dice: {dice:.4f}"
        )

        print(
            f"IoU : {iou:.4f}"
        )

    average_dice = sum(
        result["dice"]
        for result in results.values()
    ) / len(results)

    average_iou = sum(
        result["iou"]
        for result in results.values()
    ) / len(results)

    output = {
        "experiment": "FedMed Federated Learning",
        "strategy": "FedAvg",
        "device": str(DEVICE),
        "hospital_results": results,
        "average_dice": average_dice,
        "average_iou": average_iou
    }

    os.makedirs(
        "results",
        exist_ok=True
    )

    with open(
        "results/federated_evaluation.json",
        "w"
    ) as file:

        json.dump(
            output,
            file,
            indent=4
        )

    print("\n" + "=" * 60)

    print(
        f"Average Dice: {average_dice:.4f}"
    )

    print(
        f"Average IoU : {average_iou:.4f}"
    )

    print(
        "\nEvaluation saved to:"
    )

    print(
        "results/federated_evaluation.json"
    )

    print("=" * 60)


if __name__ == "__main__":
    main()