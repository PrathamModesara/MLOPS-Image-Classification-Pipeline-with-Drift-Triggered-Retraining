import mlflow

from src.config import (
    MODEL_NAME,
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    MODEL_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    DATASET_NAME,
)

from src.train import train_model
from src.evaluate import evaluate_model


def run_baseline():

    print("=" * 60)
    print("FOOD-101 MLFLOW BASELINE")
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
        run_name="deit-tiny-baseline"
    ):

        print("\nMLflow run started.")

        # ----------------------------------------------------
        # 3. Log parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "dataset",
            "Food-101"
        )

        mlflow.log_param(
            "dataset_repository",
            DATASET_NAME
        )

        mlflow.log_param(
            "model",
            MODEL_NAME
        )

        mlflow.log_param(
            "num_classes",
            NUM_CLASSES
        )

        mlflow.log_param(
            "batch_size",
            BATCH_SIZE
        )

        mlflow.log_param(
            "epochs",
            EPOCHS
        )

        mlflow.log_param(
            "learning_rate",
            LEARNING_RATE
        )

        mlflow.log_param(
            "train_images",
            505
        )

        mlflow.log_param(
            "validation_images",
            101
        )

        # ----------------------------------------------------
        # 4. Train model
        # ----------------------------------------------------

        print("\nStarting training...")

        train_model()

        # ----------------------------------------------------
        # 5. Evaluate model
        # ----------------------------------------------------

        print("\nStarting evaluation...")

        metrics = evaluate_model()

        # ----------------------------------------------------
        # 6. Log metrics
        # ----------------------------------------------------

        print("\nLogging metrics to MLflow...")

        for name, value in metrics.items():

            mlflow.log_metric(
                name,
                float(value)
            )

        # ----------------------------------------------------
        # 7. Log model artifacts
        # ----------------------------------------------------

        print("\nLogging model artifacts...")

        mlflow.log_artifacts(
            MODEL_DIR,
            artifact_path="model"
        )

        # ----------------------------------------------------
        # 8. Log useful tags
        # ----------------------------------------------------

        mlflow.set_tag(
            "stage",
            "baseline"
        )

        mlflow.set_tag(
            "framework",
            "huggingface-transformers"
        )

        mlflow.set_tag(
            "model_type",
            "DeiT-Tiny"
        )

        # ----------------------------------------------------
        # 9. Print run information
        # ----------------------------------------------------

        run_id = (
            mlflow.active_run()
            .info.run_id
        )

        print("\n" + "=" * 60)
        print("MLFLOW BASELINE COMPLETED")
        print("=" * 60)

        print(
            "Experiment:",
            MLFLOW_EXPERIMENT
        )

        print(
            "Run ID:",
            run_id
        )

        print(
            "Accuracy:",
            metrics["accuracy"]
        )

        print(
            "F1 Score:",
            metrics["f1_score"]
        )


if __name__ == "__main__":

    run_baseline()
