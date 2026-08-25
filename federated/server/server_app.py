import os

import torch

from flwr.app import ArrayRecord, Context
from flwr.serverapp import Grid, ServerApp
from flwr.serverapp.strategy import FedAvg

from models.unet import UNet3D


app = ServerApp()


@app.main()
def main(
    grid: Grid,
    context: Context,
):

    print("\n" + "=" * 60)
    print("FedMed Federated Learning Server")
    print("=" * 60)

    # --------------------------------------
    # Create initial global model
    # --------------------------------------

    global_model = UNet3D()

    arrays = ArrayRecord(
        global_model.state_dict()
    )

    # --------------------------------------
    # FedAvg strategy
    # --------------------------------------

    strategy = FedAvg(
        fraction_train=1.0,
        fraction_evaluate=1.0,
        min_train_nodes=3,
        min_evaluate_nodes=3,
        min_available_nodes=3,
    )

    num_rounds = int(
        context.run_config[
            "num-server-rounds"
        ]
    )

    print(
        f"\nFederated rounds: {num_rounds}"
    )

    print(
        "Participating hospitals: 3"
    )

    print(
        "Aggregation strategy: FedAvg"
    )

    print(
        "\nStarting federated training..."
    )

    # --------------------------------------
    # Start FedAvg
    # --------------------------------------

    result = strategy.start(
        grid=grid,
        initial_arrays=arrays,
        num_rounds=num_rounds,
    )

    # --------------------------------------
    # Save final global model
    # --------------------------------------

    print(
        "\nSaving final federated model..."
    )

    final_state_dict = (
        result.arrays.to_torch_state_dict()
    )

    checkpoint_dir = (
        "models/checkpoints"
    )

    os.makedirs(
        checkpoint_dir,
        exist_ok=True
    )

    checkpoint_path = os.path.join(
        checkpoint_dir,
        "federated_global.pt"
    )

    torch.save(
        final_state_dict,
        checkpoint_path
    )

    # --------------------------------------
    # Verify checkpoint
    # --------------------------------------

    file_size = os.path.getsize(
        checkpoint_path
    )

    print(
        f"\nGlobal model saved to:"
    )

    print(
        checkpoint_path
    )

    print(
        f"Checkpoint size: "
        f"{file_size / (1024 * 1024):.2f} MB"
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "Federated training completed!"
    )

    print("=" * 60)

    # --------------------------------------
    # Print Flower result summary
    # --------------------------------------

    print(
        "\nFederated training metrics:"
    )

    print(
        result.train_metrics_clientapp
    )

    print(
        "\nFederated evaluation metrics:"
    )

    print(
        result.evaluate_metrics_clientapp
    )