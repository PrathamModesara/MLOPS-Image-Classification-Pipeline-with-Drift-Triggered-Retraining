import os


# ============================================================
# DATASET
# ============================================================

DATASET_NAME = "ethz/food101"

DATASET_SPLIT = os.getenv(
    "FOOD101_SPLIT",
    "train[:5000]"
)

VALIDATION_SPLIT = os.getenv(
    "FOOD101_VALIDATION_SPLIT",
    "validation[:1000]"
)

NUM_CLASSES = 101


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = (
    "facebook/deit-tiny-patch16-224"
)

IMAGE_SIZE = 224


# ============================================================
# TRAINING
# ============================================================

# CPU-friendly batch size
BATCH_SIZE = int(
    os.getenv("BATCH_SIZE", "2")
)

# Final training epochs
EPOCHS = int(
    os.getenv("EPOCHS", "10")
)

# Best learning rate found by Optuna
LEARNING_RATE = float(
    os.getenv(
        "LEARNING_RATE",
        "0.000088567"
    )
)

# Best augmentation strength found by Optuna
AUGMENTATION_STRENGTH = float(
    os.getenv(
        "AUGMENTATION_STRENGTH",
        "0.069617"
    )
)


# ============================================================
# PATHS
# ============================================================

MODEL_DIR = "models/food101_model"

ARTIFACT_DIR = "artifacts"

DRIFT_REFERENCE = (
    "artifacts/reference_embeddings.npy"
)

DRIFT_CURRENT = (
    "artifacts/current_embeddings.npy"
)

DRIFT_RESULT = (
    "artifacts/drift_result.json"
)


# ============================================================
# DRIFT
# ============================================================

DRIFT_THRESHOLD = float(
    os.getenv(
        "DRIFT_THRESHOLD",
        "4.0"
    )
)


# ============================================================
# MLFLOW
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

MLFLOW_EXPERIMENT = (
    "Food101-Drift-Retraining"
)
