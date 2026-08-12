import os


# ============================================================
# DATASET
# ============================================================

DATASET_NAME = "ethz/food101"

# Development dataset
# We start with 500 images to test the pipeline.
DATASET_SPLIT = os.getenv(
    "FOOD101_SPLIT",
    "train[:500]"
)


# ============================================================
# MODEL
# ============================================================

MODEL_NAME = os.getenv(
    "MODEL_NAME",
    "facebook/deit-tiny-patch16-224"
)

NUM_CLASSES = 101

IMAGE_SIZE = 224


# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE = int(
    os.getenv("BATCH_SIZE", "8")
)

EPOCHS = int(
    os.getenv("EPOCHS", "1")
)

LEARNING_RATE = float(
    os.getenv("LEARNING_RATE", "0.0001")
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
# MLFLOW
# ============================================================

MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    "http://127.0.0.1:5000"
)

MLFLOW_EXPERIMENT = (
    "Food101-Drift-Retraining"
)


# ============================================================
# DRIFT
# ============================================================

DRIFT_THRESHOLD = float(
    os.getenv("DRIFT_THRESHOLD", "0.30")
)
