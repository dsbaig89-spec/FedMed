import json
import os


CENTRALIZED_HISTORY = "baseline/training_history.json"
FEDERATED_RESULTS = "results/federated_evaluation.json"
DP_RESULTS = "results/dp_federated_evaluation.json"

OUTPUT_FILE = "results/model_comparison.json"


def load_json(path):
    with open(path, "r") as file:
        return json.load(file)


def main():

    required_files = [
        CENTRALIZED_HISTORY,
        FEDERATED_RESULTS,
        DP_RESULTS,
    ]

    for path in required_files:
        if not os.path.exists(path):
            print(f"Missing: {path}")
            return

    centralized = load_json(CENTRALIZED_HISTORY)
    federated = load_json(FEDERATED_RESULTS)
    dp = load_json(DP_RESULTS)

    centralized_dice = max(
        centralized["validation_dice"]
    )

    centralized_iou = max(
        centralized["validation_iou"]
    )

    federated_dice = federated["average_dice"]
    federated_iou = federated["average_iou"]

    dp_dice = dp["average_dice"]
    dp_iou = dp["average_iou"]

    comparison = {

        "centralized": {
            "dice": centralized_dice,
            "iou": centralized_iou
        },

        "federated": {
            "dice": federated_dice,
            "iou": federated_iou
        },

        "federated_dp": {
            "dice": dp_dice,
            "iou": dp_iou
        },

        "difference": {
            "federated_vs_centralized_dice":
                federated_dice - centralized_dice,

            "federated_vs_centralized_iou":
                federated_iou - centralized_iou,

            "dp_vs_federated_dice":
                dp_dice - federated_dice,

            "dp_vs_federated_iou":
                dp_iou - federated_iou
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
    print("FEDMED MODEL COMPARISON")
    print("=" * 60)

    print(
        f"\nCentralized      Dice: {centralized_dice:.4f} "
        f"IoU: {centralized_iou:.4f}"
    )

    print(
        f"Federated       Dice: {federated_dice:.4f} "
        f"IoU: {federated_iou:.4f}"
    )

    print(
        f"Federated + DP  Dice: {dp_dice:.4f} "
        f"IoU: {dp_iou:.4f}"
    )

    print(f"\nSaved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()