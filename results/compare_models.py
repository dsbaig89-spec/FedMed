import json
import os


CENTRALIZED_HISTORY = (
    "baseline/training_history.json"
)

FEDERATED_RESULTS = (
    "results/federated_evaluation.json"
)

OUTPUT_FILE = (
    "results/centralized_vs_federated.json"
)


def main():

    if not os.path.exists(
        CENTRALIZED_HISTORY
    ):

        print(
            "Centralized training history not found."
        )

        return

    if not os.path.exists(
        FEDERATED_RESULTS
    ):

        print(
            "Federated evaluation not found."
        )

        return

    with open(
        CENTRALIZED_HISTORY,
        "r"
    ) as file:

        centralized = json.load(file)

    with open(
        FEDERATED_RESULTS,
        "r"
    ) as file:

        federated = json.load(file)

    centralized_dice = max(
        centralized["validation_dice"]
    )

    centralized_iou = max(
        centralized["validation_iou"]
    )

    federated_dice = (
        federated["average_dice"]
    )

    federated_iou = (
        federated["average_iou"]
    )

    comparison = {

        "centralized": {
            "dice": centralized_dice,
            "iou": centralized_iou
        },

        "federated": {
            "dice": federated_dice,
            "iou": federated_iou
        },

        "difference": {
            "dice": (
                federated_dice
                - centralized_dice
            ),

            "iou": (
                federated_iou
                - centralized_iou
            )
        }
    }

    with open(
        OUTPUT_FILE,
        "w"
    ) as file:

        json.dump(
            comparison,
            file,
            indent=4
        )

    print("=" * 60)
    print(
        "CENTRALIZED VS FEDERATED"
    )
    print("=" * 60)

    print(
        f"\nCentralized Dice : "
        f"{centralized_dice:.4f}"
    )

    print(
        f"Federated Dice  : "
        f"{federated_dice:.4f}"
    )

    print(
        f"\nCentralized IoU  : "
        f"{centralized_iou:.4f}"
    )

    print(
        f"Federated IoU   : "
        f"{federated_iou:.4f}"
    )

    print(
        "\nSaved:"
    )

    print(OUTPUT_FILE)


if __name__ == "__main__":
    main()