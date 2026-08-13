import json
import os
import subprocess

import mlflow

from src.config import (
    DRIFT_RESULT,
    DRIFT_THRESHOLD,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    LEARNING_RATE,
    BATCH_SIZE,
    EPOCHS,
    MODEL_DIR,
)


# ============================================================
# MLFLOW SETUP
# ============================================================

def setup_mlflow():

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )


# ============================================================
# LOAD DRIFT RESULT
# ============================================================

def load_drift_result():

    if not os.path.exists(
        DRIFT_RESULT
    ):

        print(
            "\nDrift result not found."
        )

        print(
            "Run the drift detector first."
        )

        return None

    with open(
        DRIFT_RESULT,
        "r",
    ) as file:

        result = json.load(file)

    return result


# ============================================================
# LOG DRIFT INFORMATION
# ============================================================

def log_drift_information(
    result,
):

    drift_score = float(
        result.get(
            "drift_score",
            0.0,
        )
    )

    threshold = float(
        result.get(
            "threshold",
            DRIFT_THRESHOLD,
        )
    )

    drift_detected = bool(
        result.get(
            "drift_detected",
            False,
        )
    )

    dataset_type = result.get(
        "dataset_type",
        "unknown",
    )

    # --------------------------------------------------------
    # Parameters
    # --------------------------------------------------------

    mlflow.log_param(
        "retraining_trigger",
        "data_drift",
    )

    mlflow.log_param(
        "drift_detected",
        str(drift_detected),
    )

    mlflow.log_param(
        "dataset_type",
        dataset_type,
    )

    mlflow.log_param(
        "drift_threshold",
        threshold,
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    mlflow.log_metric(
        "drift_score",
        drift_score,
    )

    mlflow.log_metric(
        "drift_threshold",
        threshold,
    )

    # --------------------------------------------------------
    # Artifact
    # --------------------------------------------------------

    if os.path.exists(
        DRIFT_RESULT
    ):

        mlflow.log_artifact(
            DRIFT_RESULT,
            artifact_path="drift",
        )

    print(
        "\nMLflow drift information logged."
    )

    print(
        f"Drift score: {drift_score:.4f}"
    )

    print(
        f"Drift threshold: {threshold:.4f}"
    )

    print(
        f"Drift detected: {drift_detected}"
    )


# ============================================================
# LOG TRAINING PARAMETERS
# ============================================================

def log_training_parameters():

    mlflow.log_param(
        "learning_rate",
        LEARNING_RATE,
    )

    mlflow.log_param(
        "batch_size",
        BATCH_SIZE,
    )

    mlflow.log_param(
        "epochs",
        EPOCHS,
    )

    print(
        "\nMLflow training parameters logged."
    )

    print(
        f"Learning rate: {LEARNING_RATE}"
    )

    print(
        f"Batch size: {BATCH_SIZE}"
    )

    print(
        f"Epochs: {EPOCHS}"
    )


# ============================================================
# LOG MODEL ARTIFACT
# ============================================================

def log_model_artifact():

    if not os.path.exists(
        MODEL_DIR
    ):

        print(
            "\nModel directory not found:"
        )

        print(
            MODEL_DIR
        )

        return False

    print(
        "\nLogging retrained model..."
    )

    mlflow.log_artifacts(
        MODEL_DIR,
        artifact_path="retrained_model",
    )

    print(
        "Retrained model artifacts logged."
    )

    return True


# ============================================================
# RETRAINING TRIGGER
# ============================================================

def check_drift():

    print("=" * 60)
    print(
        "DRIFT TRIGGER CHECK"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Load drift result
    # --------------------------------------------------------

    result = load_drift_result()

    if result is None:

        return False

    drift_score = float(
        result.get(
            "drift_score",
            0.0,
        )
    )

    threshold = float(
        result.get(
            "threshold",
            DRIFT_THRESHOLD,
        )
    )

    drift_detected = bool(
        result.get(
            "drift_detected",
            False,
        )
    )

    print(
        f"\nDrift Score : "
        f"{drift_score:.4f}"
    )

    print(
        f"Threshold   : "
        f"{threshold:.4f}"
    )

    print(
        f"Drift       : "
        f"{drift_detected}"
    )

    # ========================================================
    # DRIFT DETECTED
    # ========================================================

    if drift_detected:

        print(
            "\n" + "!" * 60
        )

        print(
            "DRIFT DETECTED"
        )

        print(
            "Starting model retraining..."
        )

        print(
            "!" * 60
        )

        # ----------------------------------------------------
        # Setup MLflow
        # ----------------------------------------------------

        setup_mlflow()

        # ----------------------------------------------------
        # Start MLflow run
        # ----------------------------------------------------

        with mlflow.start_run(
            run_name="food101-drift-retraining"
        ) as run:

            print(
                "\nMLflow run started."
            )

            print(
                f"Run ID: {run.info.run_id}"
            )

            # ------------------------------------------------
            # Log drift information
            # ------------------------------------------------

            log_drift_information(
                result
            )

            # ------------------------------------------------
            # Log training configuration
            # ------------------------------------------------

            log_training_parameters()

            # ------------------------------------------------
            # Run ZenML pipeline
            # ------------------------------------------------

            print(
                "\nStarting ZenML retraining pipeline..."
            )

            pipeline_result = subprocess.run(
                [
                    "python",
                    "-m",
                    "src.zenml_pipeline",
                ],
                check=False,
            )

            # ------------------------------------------------
            # Check pipeline result
            # ------------------------------------------------

            if pipeline_result.returncode == 0:

                print(
                    "\nZenML retraining completed successfully."
                )

                mlflow.log_param(
                    "retraining_status",
                    "completed",
                )

                # --------------------------------------------
                # Log model
                # --------------------------------------------

                model_logged = (
                    log_model_artifact()
                )

                if model_logged:

                    mlflow.log_param(
                        "model_artifact",
                        "logged",
                    )

                print(
                    "\n" + "=" * 60
                )

                print(
                    "RETRAINING COMPLETED"
                )

                print(
                    "=" * 60
                )

                print(
                    f"\nMLflow Run ID: "
                    f"{run.info.run_id}"
                )

                print(
                    "MLflow run completed."
                )

                return True

            else:

                print(
                    "\nZenML retraining failed."
                )

                mlflow.log_param(
                    "retraining_status",
                    "failed",
                )

                print(
                    "\n" + "=" * 60
                )

                print(
                    "RETRAINING FAILED"
                )

                print(
                    "=" * 60
                )

                return False

    # ========================================================
    # NO DRIFT
    # ========================================================

    print(
        "\n" + "=" * 60
    )

    print(
        "NO DRIFT DETECTED"
    )

    print(
        "Retraining is not required."
    )

    print(
        "=" * 60
    )

    return False


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    check_drift()
