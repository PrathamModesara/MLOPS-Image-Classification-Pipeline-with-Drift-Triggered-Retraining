from zenml import pipeline, step

from src.config import (
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    DRIFT_THRESHOLD,
)


# ============================================================
# DATA STEP
# ============================================================

@step(enable_cache=False)
def data_step():

    from src.data import (
        load_food101,
        create_train_validation_split,
    )

    print("=" * 60)
    print("ZENML DATA STEP")
    print("=" * 60)

    dataset = load_food101()

    train_dataset, validation_dataset = (
        create_train_validation_split(dataset)
    )

    train_size = len(train_dataset)
    validation_size = len(validation_dataset)

    train_classes = len(
        set(train_dataset["label"])
    )

    validation_classes = len(
        set(validation_dataset["label"])
    )

    print(
        f"Training samples: {train_size}"
    )

    print(
        f"Validation samples: {validation_size}"
    )

    print(
        f"Training classes: {train_classes}"
    )

    print(
        f"Validation classes: {validation_classes}"
    )

    return {
        "train_size": train_size,
        "validation_size": validation_size,
        "train_classes": train_classes,
        "validation_classes": validation_classes,
    }


# ============================================================
# DRIFT INFORMATION STEP
# ============================================================

@step(enable_cache=False)
def drift_info_step():

    import json
    import os

    print("=" * 60)
    print("ZENML DRIFT INFORMATION STEP")
    print("=" * 60)

    drift_file = "artifacts/drift_result.json"

    if not os.path.exists(drift_file):

        print(
            "Drift result file not found."
        )

        return {
            "drift_score": 0.0,
            "drift_threshold": DRIFT_THRESHOLD,
            "drift_detected": False,
        }

    with open(
        drift_file,
        "r"
    ) as file:

        result = json.load(file)

    drift_score = float(
        result.get(
            "drift_score",
            0.0
        )
    )

    threshold = float(
        result.get(
            "threshold",
            DRIFT_THRESHOLD
        )
    )

    drift_detected = bool(
        result.get(
            "drift_detected",
            False
        )
    )

    print(
        f"Drift score     : "
        f"{drift_score:.4f}"
    )

    print(
        f"Drift threshold : "
        f"{threshold:.4f}"
    )

    print(
        f"Drift detected  : "
        f"{drift_detected}"
    )

    if drift_detected:

        print(
            "\n⚠️ DRIFT DETECTED"
        )

        print(
            "This pipeline execution "
            "represents drift-triggered "
            "retraining."
        )

    else:

        print(
            "\n✅ NO SIGNIFICANT DRIFT"
        )

    return {
        "drift_score": drift_score,
        "drift_threshold": threshold,
        "drift_detected": drift_detected,
    }


# ============================================================
# TRAINING STEP
# ============================================================

@step(enable_cache=False)
def training_step(
    data_info,
    drift_info,
):

    from src.train import train_model

    print("=" * 60)
    print("ZENML TRAINING STEP")
    print("=" * 60)

    print(
        f"Training samples : "
        f"{data_info['train_size']}"
    )

    print(
        f"Validation samples : "
        f"{data_info['validation_size']}"
    )

    print(
        f"Training classes : "
        f"{data_info['train_classes']}"
    )

    print(
        f"Drift score : "
        f"{drift_info['drift_score']:.4f}"
    )

    print(
        f"Drift detected : "
        f"{drift_info['drift_detected']}"
    )

    print(
        f"Learning rate : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Batch size : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs : "
        f"{EPOCHS}"
    )

    print(
        "\nStarting model training..."
    )

    train_model()

    print(
        "\nTraining completed."
    )

    return {
        "training_status": "completed",
        "learning_rate": LEARNING_RATE,
        "batch_size": BATCH_SIZE,
        "epochs": EPOCHS,
        "drift_score": drift_info["drift_score"],
        "drift_detected": drift_info["drift_detected"],
    }


# ============================================================
# EVALUATION STEP
# ============================================================

@step(enable_cache=False)
def evaluation_step(
    training_info,
):

    from src.evaluate import evaluate_model

    print("=" * 60)
    print("ZENML EVALUATION STEP")
    print("=" * 60)

    print(
        f"Training status : "
        f"{training_info['training_status']}"
    )

    print(
        f"Retraining drift score : "
        f"{training_info['drift_score']:.4f}"
    )

    print(
        f"Retraining triggered : "
        f"{training_info['drift_detected']}"
    )

    print(
        "\nRunning model evaluation..."
    )

    metrics = evaluate_model()

    print(
        "\nEvaluation metrics:"
    )

    for name, value in metrics.items():

        print(
            f"{name}: {value:.4f}"
        )

    return metrics


# ============================================================
# ZENML PIPELINE
# ============================================================

@pipeline(
    name="food101_pipeline",
    enable_cache=False,
)
def food101_pipeline():

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    data_info = data_step()

    # --------------------------------------------------------
    # 2. Read drift information
    # --------------------------------------------------------

    drift_info = drift_info_step()

    # --------------------------------------------------------
    # 3. Train model
    #
    # training_step depends on both:
    # data_info
    # drift_info
    # --------------------------------------------------------

    training_info = training_step(
        data_info=data_info,
        drift_info=drift_info,
    )

    # --------------------------------------------------------
    # 4. Evaluate trained model
    #
    # evaluation_step depends on training_info
    # --------------------------------------------------------

    evaluation_step(
        training_info=training_info
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    print("=" * 60)
    print("FOOD-101 ZENML PIPELINE")
    print("=" * 60)

    print(
        "\nPipeline configuration:"
    )

    print(
        f"Learning rate   : "
        f"{LEARNING_RATE}"
    )

    print(
        f"Batch size      : "
        f"{BATCH_SIZE}"
    )

    print(
        f"Epochs          : "
        f"{EPOCHS}"
    )

    print(
        f"Drift threshold : "
        f"{DRIFT_THRESHOLD}"
    )

    print(
        "\nStarting ZenML pipeline..."
    )

    food101_pipeline()

    print(
        "\nZenML pipeline execution completed."
    )
