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

from privacy.differential_privacy import (
    DifferentialPrivacy,
)

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

    dp_max_norm = float(
        context.run_config.get(
            "dp-max-norm",
            1.0
        )
    )

    dp_noise_multiplier = float(
        context.run_config.get(
            "dp-noise-multiplier",
            0.01
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

    # Keep a copy of the global parameters.
    # DP will be applied to the local model
    # update relative to these parameters.

    global_state = {
        key: value.detach().clone()
        for key, value in global_state.items()
    }

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
    # Calculate local model update
    # --------------------------------------

    local_state = model.state_dict()

    update = []

    parameter_keys = []

    for key in global_state:

        if not torch.is_floating_point(
            global_state[key]
        ):
            continue

        update.append(
            (
                local_state[key].detach()
                - global_state[key]
            )
        )

        parameter_keys.append(key)

    # --------------------------------------
    # Differential Privacy
    # --------------------------------------

    dp = DifferentialPrivacy(
        max_norm=dp_max_norm,
        noise_multiplier=dp_noise_multiplier,
    )

    protected_update = dp.protect(
        update
    )

    print(
        f"[HOSPITAL {hospital_id}] "
        f"Differential Privacy applied"
    )

    print(
        f"[HOSPITAL {hospital_id}] "
        f"DP max norm={dp_max_norm}"
    )

    print(
        f"[HOSPITAL {hospital_id}] "
        f"DP noise multiplier="
        f"{dp_noise_multiplier}"
    )

    # --------------------------------------
    # Reconstruct protected model
    # --------------------------------------

    protected_state = {
        key: value.detach().clone()
        for key, value in local_state.items()
    }

    for key, protected in zip(
        parameter_keys,
        protected_update
    ):

        protected_state[key] = (
            global_state[key]
            + protected
        )

    # --------------------------------------
    # Send protected model update
    # --------------------------------------

    arrays = ArrayRecord(
        protected_state
    )

    metrics = MetricRecord(
        {
            "train_loss": float(
                train_loss
            ),
            "dice": float(
                dice
            ),
            "iou": float(
                iou
            ),
            "num-examples": len(
                dataset
            ),
            "dp_enabled": 1.0,
            "dp_max_norm": dp_max_norm,
            "dp_noise_multiplier":
                dp_noise_multiplier,
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
        f"Sending protected model update..."
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
            "num-examples": len(
                dataset
            ),
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