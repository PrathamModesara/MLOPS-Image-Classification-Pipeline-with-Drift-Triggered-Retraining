import mlflow
import mlflow.transformers

from transformers import (
    AutoModelForImageClassification,
    AutoImageProcessor,
)

from src.config import (
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
    MODEL_DIR,
    LEARNING_RATE,
    AUGMENTATION_STRENGTH,
    BATCH_SIZE,
    EPOCHS,
)


# ============================================================
# CONFIGURATION
# ============================================================

REGISTERED_MODEL_NAME = "Food101Classifier"


# ============================================================
# MODEL REGISTRATION
# ============================================================

def register_model():

    print("=" * 60)
    print("MLFLOW - FOOD-101 MODEL VERSIONING")
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

    print(
        f"\nTracking URI: "
        f"{MLFLOW_TRACKING_URI}"
    )

    print(
        f"Experiment: "
        f"{MLFLOW_EXPERIMENT}"
    )

    print(
        f"Model directory: "
        f"{MODEL_DIR}"
    )

    # --------------------------------------------------------
    # 2. Load trained model
    # --------------------------------------------------------

    print(
        "\nLoading trained vision model..."
    )

    model = (
        AutoModelForImageClassification
        .from_pretrained(
            MODEL_DIR
        )
    )

    print(
        "Vision model loaded."
    )

    # --------------------------------------------------------
    # 3. Load image processor
    # --------------------------------------------------------

    print(
        "\nLoading image processor..."
    )

    processor = (
        AutoImageProcessor
        .from_pretrained(
            MODEL_DIR
        )
    )

    print(
        "Image processor loaded."
    )

    # --------------------------------------------------------
    # 4. Create MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="food101-final-optimized-model"
    ) as run:

        print(
            "\nMLflow run started."
        )

        # ----------------------------------------------------
        # Training parameters
        # ----------------------------------------------------

        mlflow.log_param(
            "model_name",
            "facebook/deit-tiny-patch16-224",
        )

        mlflow.log_param(
            "training_type",
            "final_optuna_optimized",
        )

        mlflow.log_param(
            "learning_rate",
            LEARNING_RATE,
        )

        mlflow.log_param(
            "augmentation_strength",
            AUGMENTATION_STRENGTH,
        )

        mlflow.log_param(
            "batch_size",
            BATCH_SIZE,
        )

        mlflow.log_param(
            "epochs",
            EPOCHS,
        )

        mlflow.log_param(
            "num_classes",
            101,
        )

        # ----------------------------------------------------
        # Evaluation metrics
        # ----------------------------------------------------

        mlflow.log_metric(
            "validation_loss",
            1.8010,
        )

        mlflow.log_metric(
            "accuracy",
            0.5416,
        )

        mlflow.log_metric(
            "precision",
            0.5537,
        )

        mlflow.log_metric(
            "recall",
            0.5416,
        )

        mlflow.log_metric(
            "f1_score",
            0.5363,
        )

        # ----------------------------------------------------
        # Vision model components
        # ----------------------------------------------------

        transformers_components = {
            "model": model,
            "image_processor": processor,
        }

        print(
            "\nLogging vision model to MLflow..."
        )

        # ----------------------------------------------------
        # Log + register
        # ----------------------------------------------------

        model_info = (
            mlflow.transformers.log_model(
                transformers_model=
                    transformers_components,

                name="model",

                task="image-classification",

                registered_model_name=
                    REGISTERED_MODEL_NAME,

                save_pretrained=True,
            )
        )

        # ----------------------------------------------------
        # Success
        # ----------------------------------------------------

        print(
            "\n" + "=" * 60
        )

        print(
            "MODEL VERSIONING COMPLETED"
        )

        print(
            "=" * 60
        )

        print(
            f"\nRegistered model: "
            f"{REGISTERED_MODEL_NAME}"
        )

        print(
            f"\nModel URI: "
            f"{model_info.model_uri}"
        )

        print(
            f"\nRun ID: "
            f"{run.info.run_id}"
        )

        print(
            "\nMLflow UI:"
        )

        print(
            "http://127.0.0.1:5000"
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    register_model()
