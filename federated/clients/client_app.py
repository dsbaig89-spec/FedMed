
import torch

from flwr.app import (
    ArrayRecord,
    Context,
    Message,
    MetricRecord,
    RecordDict,
)

from flwr.clientapp import ClientApp

from models.unet import UNet3D

from federated.common.federated_utils import (
    DEVICE,
    load_hospital_data,
    train_local_model,
    evaluate_local_model,
)


app = ClientApp()


HOSPITALS = {
    0: "A",
    1: "B",
    2: "C",
}


@app.train()
def train(
    msg: Message,
    context: Context,
) -> Message:

    # --------------------------------------
    # Identify hospital
    # --------------------------------------

    partition_id = int(
        context.node_config["partition-id"]
    )

    hospital_id = HOSPITALS.get(
        partition_id,
        "A"
    )

    print(
        f"\n[HOSPITAL {hospital_id}] "
        f"Starting local training..."
    )

    # --------------------------------------
    # Read configuration
    # --------------------------------------

    local_epochs = int(
        context.run_config.get(
            "local-epochs",
            1
        )
    )

    learning_rate = float(
        context.run_config.get(
            "learning-rate",
            0.001
        )
    )

    # --------------------------------------
    # Load PRIVATE hospital dataset
    # --------------------------------------

    dataset, train_loader = load_hospital_data(
        hospital_id
    )

    print(
        f"[HOSPITAL {hospital_id}] "
        f"Private samples: {len(dataset)}"
    )

    # --------------------------------------
    # Create local model
    # --------------------------------------

    model = UNet3D().to(DEVICE)

    # --------------------------------------
    # Receive global model
    # --------------------------------------

    global_state = (
        msg.content[
            "arrays"
        ].to_torch_state_dict()
    )

    model.load_state_dict(
        global_state
    )

    # --------------------------------------
    # Local training
    # --------------------------------------

    train_loss = train_local_model(
        model=model,
        train_loader=train_loader,
        epochs=local_epochs,
        learning_rate=learning_rate,
    )

    # --------------------------------------
    # Evaluate local model
    # --------------------------------------

    val_loss, dice, iou = (
        evaluate_local_model(
            model,
            train_loader
        )
    )

    print(
        f"[HOSPITAL {hospital_id}] "
        f"Loss={train_loss:.4f} "
        f"Dice={dice:.4f} "
        f"IoU={iou:.4f}"
    )

    # --------------------------------------
    # Send model update
    # --------------------------------------

    arrays = ArrayRecord(
        model.state_dict()
    )

    metrics = MetricRecord(
        {
            "train_loss": float(
                train_loss
            ),
            "dice": float(dice),
            "iou": float(iou),
            "num-examples": len(dataset),
        }
    )

    content = RecordDict(
        {
            "arrays": arrays,
            "metrics": metrics,
        }
    )

    print(
        f"[HOSPITAL {hospital_id}] "
        f"Sending model update..."
    )

    return Message(
        content=content,
        reply_to=msg,
    )


@app.evaluate()
def evaluate(
    msg: Message,
    context: Context,
) -> Message:

    partition_id = int(
        context.node_config["partition-id"]
    )

    hospital_id = HOSPITALS.get(
        partition_id,
        "A"
    )

    dataset, loader = load_hospital_data(
        hospital_id
    )

    model = UNet3D().to(DEVICE)

    state_dict = (
        msg.content[
            "arrays"
        ].to_torch_state_dict()
    )

    model.load_state_dict(
        state_dict
    )

    loss, dice, iou = (
        evaluate_local_model(
            model,
            loader
        )
    )

    print(
        f"[HOSPITAL {hospital_id}] "
        f"Evaluation: "
        f"Dice={dice:.4f}, "
        f"IoU={iou:.4f}"
    )

    metrics = MetricRecord(
        {
            "loss": float(loss),
            "dice": float(dice),
            "iou": float(iou),
            "num-examples": len(dataset),
        }
    )

    return Message(
        content=RecordDict(
            {
                "metrics": metrics
            }
        ),
        reply_to=msg,
    )
