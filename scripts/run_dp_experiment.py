import json
from pathlib import Path

import torch

from models.unet import UNet3D
from privacy.differential_privacy import DifferentialPrivacy

from federated.common.federated_utils import (
    DEVICE,
    load_hospital_data,
    train_local_model,
    evaluate_local_model,
)


HOSPITALS = ["A", "B", "C"]

LOCAL_EPOCHS = 1
LEARNING_RATE = 0.001

DP_MAX_NORM = 1.0
DP_NOISE_MULTIPLIER = 0.01

ROUNDS = 3

CHECKPOINT_DIR = Path("models/checkpoints")
RESULTS_DIR = Path("results")

CHECKPOINT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


def fedavg(
    global_state,
    local_states,
    sample_counts,
):
    """
    Weighted Federated Averaging.
    """

    total_samples = sum(sample_counts)

    new_state = {}

    for key in global_state:

        if not torch.is_floating_point(
            global_state[key]
        ):
            new_state[key] = global_state[key]
            continue

        aggregated = torch.zeros_like(
            global_state[key]
        )

        for local_state, count in zip(
            local_states,
            sample_counts
        ):

            weight = count / total_samples

            aggregated += (
                local_state[key] * weight
            )

        new_state[key] = aggregated

    return new_state


def create_dp_update(
    global_state,
    local_state,
):
    """
    Create a local model update,
    apply DP clipping + Gaussian noise,
    then reconstruct the protected model.
    """

    update = []
    keys = []

    for key in global_state:

        if not torch.is_floating_point(
            global_state[key]
        ):
            continue

        update.append(
            local_state[key].detach()
            - global_state[key]
        )

        keys.append(key)

    dp = DifferentialPrivacy(
        max_norm=DP_MAX_NORM,
        noise_multiplier=DP_NOISE_MULTIPLIER,
    )

    protected_update = dp.protect(
        update
    )

    protected_state = {
        key: value.detach().clone()
        for key, value in local_state.items()
    }

    for key, protected in zip(
        keys,
        protected_update
    ):

        protected_state[key] = (
            global_state[key]
            + protected
        )

    return protected_state


def main():

    print("=" * 60)
    print("FedMed Low-Memory Differential Privacy Experiment")
    print("=" * 60)

    print(f"\nDevice: {DEVICE}")
    print(f"Hospitals: {len(HOSPITALS)}")
    print(f"Rounds: {ROUNDS}")
    print(f"Local epochs: {LOCAL_EPOCHS}")
    print(f"Learning rate: {LEARNING_RATE}")
    print(f"DP max norm: {DP_MAX_NORM}")
    print(
        f"DP noise multiplier: "
        f"{DP_NOISE_MULTIPLIER}"
    )

    # --------------------------------------
    # Create initial global model
    # --------------------------------------

    global_model = UNet3D().to(DEVICE)

    global_state = {
        key: value.detach().clone()
        for key, value in
        global_model.state_dict().items()
    }

    history = []

    # --------------------------------------
    # Federated rounds
    # --------------------------------------

    for round_number in range(
        1,
        ROUNDS + 1
    ):

        print("\n" + "-" * 60)
        print(
            f"FEDERATED ROUND {round_number}/{ROUNDS}"
        )
        print("-" * 60)

        local_states = []
        sample_counts = []

        round_metrics = []

        for hospital_id in HOSPITALS:

            print(
                f"\n[HOSPITAL {hospital_id}] "
                f"Starting local training..."
            )

            dataset, train_loader = (
                load_hospital_data(
                    hospital_id
                )
            )

            print(
                f"[HOSPITAL {hospital_id}] "
                f"Samples: {len(dataset)}"
            )

            # Create local model
            local_model = (
                UNet3D().to(DEVICE)
            )

            local_model.load_state_dict(
                global_state
            )

            # Local training
            train_loss = train_local_model(
                model=local_model,
                train_loader=train_loader,
                epochs=LOCAL_EPOCHS,
                learning_rate=LEARNING_RATE,
            )

            # Local evaluation
            val_loss, dice, iou = (
                evaluate_local_model(
                    local_model,
                    train_loader
                )
            )

            print(
                f"[HOSPITAL {hospital_id}] "
                f"Loss={train_loss:.4f} "
                f"Dice={dice:.4f} "
                f"IoU={iou:.4f}"
            )

            # Apply DP
            protected_state = (
                create_dp_update(
                    global_state,
                    local_model.state_dict()
                )
            )

            print(
                f"[HOSPITAL {hospital_id}] "
                f"✓ Differential Privacy applied"
            )

            local_states.append(
                protected_state
            )

            sample_counts.append(
                len(dataset)
            )

            round_metrics.append(
                {
                    "hospital": hospital_id,
                    "train_loss": float(
                        train_loss
                    ),
                    "dice": float(dice),
                    "iou": float(iou),
                    "samples": len(dataset),
                }
            )

            # Free local model memory
            del local_model
            del dataset
            del train_loader

        # --------------------------------------
        # FedAvg
        # --------------------------------------

        print("\nAggregating protected updates...")

        global_state = fedavg(
            global_state,
            local_states,
            sample_counts,
        )

        print("✓ FedAvg aggregation completed")

        # Free local states
        del local_states

        # --------------------------------------
        # Round metrics
        # --------------------------------------

        avg_dice = sum(
            item["dice"]
            for item in round_metrics
        ) / len(round_metrics)

        avg_iou = sum(
            item["iou"]
            for item in round_metrics
        ) / len(round_metrics)

        history.append(
            {
                "round": round_number,
                "average_dice": avg_dice,
                "average_iou": avg_iou,
                "hospitals": round_metrics,
            }
        )

        print(
            f"\nRound {round_number} summary:"
        )

        print(
            f"Average Dice: {avg_dice:.4f}"
        )

        print(
            f"Average IoU : {avg_iou:.4f}"
        )

    # --------------------------------------
    # Save final model
    # --------------------------------------

    checkpoint_path = (
        CHECKPOINT_DIR
        / "dp_federated_global.pt"
    )

    torch.save(
        global_state,
        checkpoint_path
    )

    print("\n" + "=" * 60)
    print("DP FEDERATED TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"\nCheckpoint: {checkpoint_path}"
    )

    print(
        f"Checkpoint size: "
        f"{checkpoint_path.stat().st_size / 1024 / 1024:.2f} MB"
    )

    # --------------------------------------
    # Save experiment history
    # --------------------------------------

    result = {
        "experiment": "Federated Learning with Differential Privacy",
        "hospitals": HOSPITALS,
        "rounds": ROUNDS,
        "local_epochs": LOCAL_EPOCHS,
        "learning_rate": LEARNING_RATE,
        "dp_max_norm": DP_MAX_NORM,
        "dp_noise_multiplier": DP_NOISE_MULTIPLIER,
        "history": history,
        "checkpoint": str(
            checkpoint_path
        ),
    }

    result_path = (
        RESULTS_DIR
        / "dp_federated_history.json"
    )

    with open(
        result_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            result,
            file,
            indent=4
        )

    print(
        f"History: {result_path}"
    )

    print("\n✓ Experiment completed successfully")


if __name__ == "__main__":
    main()