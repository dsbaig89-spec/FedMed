from fastapi import FastAPI

app = FastAPI(
    title="FedMed API",
    description="Federated Medical Image Segmentation Backend",
    version="1.0.0",
)


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": "FedMed API",
    }

import json
from pathlib import Path


@app.get("/results")
def get_results():
    files = {
        "centralized": "baseline/training_history.json",
        "federated": "results/federated_evaluation.json",
        "federated_dp": "results/dp_federated_evaluation.json",
        "comparison": "results/model_comparison.json",
    }

    output = {}

    for name, file_path in files.items():
        path = Path(file_path)

        if path.exists():
            with open(path, "r") as file:
                output[name] = json.load(file)
        else:
            output[name] = None

    return output

HOSPITALS = {
    "Hospital A": "datasets/hospital_a",
    "Hospital B": "datasets/hospital_b",
    "Hospital C": "datasets/hospital_c",
}


@app.get("/hospitals")
def get_hospitals():
    output = {}

    for hospital, path in HOSPITALS.items():
        dataset_path = Path(path)

        samples = len(list(dataset_path.glob("*"))) if dataset_path.exists() else 0

        output[hospital] = {
            "status": "available" if samples > 0 else "unavailable",
            "samples": samples,
        }

    return output
@app.get("/training/status")
def training_status():
    return {
        "status": "completed",
        "rounds": 3,
        "hospitals": 3,
        "device": "cpu",
        "federated": True,
        "differential_privacy": True,
    } 
@app.post("/training/start")
def start_training():
    return {
        "message": "Training request received",
        "status": "ready",
        "note": "Training can be started from the FedMed training pipeline."
    }