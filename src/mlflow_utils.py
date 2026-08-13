import os
import json

import mlflow

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    MODEL_DIR,
)


def setup_mlflow():

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )


def log_retraining_parameters(
    drift_score,
    drift_detected,
):

    mlflow.log_param(
        "retraining_trigger",
        "drift",
    )

    mlflow.log_param(
        "drift_detected",
        drift_detected,
    )

    mlflow.log_metric(
        "drift_score",
        drift_score,
    )


def log_training_parameters(
    learning_rate,
    batch_size,
    epochs,
):

    mlflow.log_param(
        "learning_rate",
        learning_rate,
    )

    mlflow.log_param(
        "batch_size",
        batch_size,
    )

    mlflow.log_param(
        "epochs",
        epochs,
    )


def log_model_artifact(
    model_dir,
):

    if os.path.exists(model_dir):

        mlflow.log_artifacts(
            model_dir,
            artifact_path="model",
        )


def log_drift_result(
    drift_result_path="artifacts/drift_result.json",
):

    if not os.path.exists(
        drift_result_path
    ):
        print(
            "Drift result file not found:"
        )
        print(
            drift_result_path
        )
        return

    with open(
        drift_result_path,
        "r",
    ) as file:

        result = json.load(file)

    drift_score = result.get(
        "drift_score",
        0.0,
    )

    threshold = result.get(
        "threshold",
        0.0,
    )

    drift_detected = result.get(
        "drift_detected",
        False,
    )

    mlflow.log_metric(
        "drift_score",
        float(drift_score),
    )

    mlflow.log_metric(
        "drift_threshold",
        float(threshold),
    )

    mlflow.log_param(
        "drift_detected",
        str(drift_detected),
    )

    mlflow.log_param(
        "dataset_type",
        result.get(
            "dataset_type",
            "unknown",
        ),
    )

    mlflow.log_artifact(
        drift_result_path,
        artifact_path="drift",
    )


def log_retrained_model():

    if os.path.exists(
        MODEL_DIR
    ):

        mlflow.log_artifacts(
            MODEL_DIR,
            artifact_path="retrained_model",
        )

        print(
            "Retrained model artifacts logged."
        )

    else:

        print(
            "Retrained model directory "
            "not found:"
        )

        print(
            MODEL_DIR
        )
