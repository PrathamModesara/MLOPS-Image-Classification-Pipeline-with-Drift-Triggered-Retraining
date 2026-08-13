import os

import mlflow

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    MODEL_DIR,
)


def main():

    print("=" * 60)
    print("MLFLOW - FOOD-101 IMPROVED BASELINE")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Configure MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    # --------------------------------------------------------
    # 2. Start MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="food101-improved-baseline"
    ):

        print("\nLogging parameters...")

        # Parameters
        mlflow.log_params({
            "model_name":
                "facebook/deit-tiny-patch16-224",

            "training_samples":
                5050,

            "validation_samples":
                1010,

            "training_classes":
                101,

            "validation_classes":
                101,

            "epochs":
                10,

            "batch_size":
                2,

            "learning_rate":
                0.001,
        })

        # ----------------------------------------------------
        # 3. Metrics
        # ----------------------------------------------------

        print("Logging metrics...")

        mlflow.log_metrics({
            "validation_loss":
                2.1555,

            "accuracy":
                0.5693,

            "precision":
                0.5837,

            "recall":
                0.5693,

            "f1_score":
                0.5634,
        })

        # ----------------------------------------------------
        # 4. Log baseline metrics file
        # ----------------------------------------------------

        metrics_file = (
            "artifacts/baseline_metrics.json"
        )

        if os.path.exists(metrics_file):

            mlflow.log_artifact(
                metrics_file,
                artifact_path="metrics",
            )

            print(
                "Baseline metrics artifact logged."
            )

        # ----------------------------------------------------
        # 5. Log model
        # ----------------------------------------------------

        if os.path.exists(MODEL_DIR):

            print(
                "\nLogging trained model..."
            )

            mlflow.log_artifacts(
                MODEL_DIR,
                artifact_path="model",
            )

            print(
                "Model artifacts logged."
            )

        else:

            print(
                "\nWARNING: Model directory not found:"
            )

            print(MODEL_DIR)

        # ----------------------------------------------------
        # 6. Print run information
        # ----------------------------------------------------

        run = mlflow.active_run()

        print("\n" + "=" * 60)
        print("MLFLOW BASELINE LOGGED")
        print("=" * 60)

        print(
            f"Run ID: {run.info.run_id}"
        )

        print(
            f"Experiment ID: "
            f"{run.info.experiment_id}"
        )

        print(
            f"Tracking URI: "
            f"{MLFLOW_TRACKING_URI}"
        )

        print(
            "\nOpen MLflow UI:"
        )

        print(
            "http://127.0.0.1:5000"
        )


if __name__ == "__main__":

    main()
