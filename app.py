import os
import time

import torch
from flask import Flask, jsonify, request
from PIL import Image

from prometheus_client import (
    Counter,
    Gauge,
    Histogram,
    generate_latest,
    CONTENT_TYPE_LATEST,
)

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models",
    "food101_model",
)

DEVICE = torch.device("cpu")

# Reduce CPU thread usage on Render Free.
torch.set_num_threads(1)


# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)


# ============================================================
# GLOBAL MODEL VARIABLES
# ============================================================

# IMPORTANT:
# The model is NOT loaded when app.py is imported.
#
# This allows Gunicorn to start quickly and Render to detect
# the PORT before the large Food-101 model is loaded.

processor = None
model = None


# ============================================================
# PROMETHEUS METRICS
# ============================================================

prediction_requests = Counter(
    "food101_prediction_requests_total",
    "Total number of prediction requests received",
)

prediction_success = Counter(
    "food101_prediction_success_total",
    "Total number of successful Food-101 predictions",
)

prediction_errors = Counter(
    "food101_prediction_errors_total",
    "Total number of failed Food-101 predictions",
)

prediction_confidence = Gauge(
    "food101_prediction_confidence",
    "Confidence of the most recent Food-101 prediction",
)

prediction_class_id = Gauge(
    "food101_prediction_class_id",
    "Class ID of the most recent Food-101 prediction",
)

prediction_latency = Histogram(
    "food101_prediction_latency_seconds",
    "Prediction request processing time in seconds",
)


# ============================================================
# MODEL LOADING FUNCTION
# ============================================================

def load_model():
    """
    Load the Food-101 image processor and trained model.

    The model is loaded only when the first prediction request
    arrives. This prevents Render's port scanner from timing
    out while the model is being loaded.
    """

    global processor
    global model

    # If already loaded, do nothing.
    if model is not None and processor is not None:
        return

    print("=" * 60)
    print("LOADING FOOD-101 MODEL")
    print("=" * 60)

    print("Model directory:", MODEL_DIR)
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Load image processor
    # --------------------------------------------------------

    print("Loading image processor...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_DIR
    )

    print("Image processor loaded.")

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print("Loading trained Food-101 model...")

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_DIR,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )

    # Move model to CPU.
    model.to(DEVICE)

    # Evaluation mode.
    model.eval()

    print("Trained Food-101 model loaded.")
    print(
        "Number of classes:",
        model.config.num_labels,
    )

    print("=" * 60)


# ============================================================
# ROOT ENDPOINT
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify(
        {
            "message": "Food-101 Image Classification API",
            "status": "running",
            "endpoints": {
                "health": "/health",
                "model_info": "/model-info",
                "prediction": "POST /predict",
                "metrics": "/metrics",
            },
        }
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify(
        {
            "status": "ok",
            "service": "Food-101 Inference API",
            "model_loaded": model is not None,
            "device": str(DEVICE),
        }
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.route("/model-info", methods=["GET"])
def model_info():

    if model is not None:

        num_classes = model.config.num_labels

    else:

        num_classes = 101

    return jsonify(
        {
            "model": "Food-101 trained DeiT-Tiny",
            "num_classes": num_classes,
            "device": str(DEVICE),
            "model_loaded": model is not None,
            "model_directory": "models/food101_model",
        }
    )


# ============================================================
# PROMETHEUS METRICS
# ============================================================

@app.route("/metrics", methods=["GET"])
def metrics():

    return generate_latest(), 200, {
        "Content-Type": CONTENT_TYPE_LATEST,
    }


# ============================================================
# PREDICTION ENDPOINT
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    start_time = time.time()

    prediction_requests.inc()

    # --------------------------------------------------------
    # Check uploaded image
    # --------------------------------------------------------

    if "image" not in request.files:

        prediction_errors.inc()

        prediction_latency.observe(
            time.time() - start_time
        )

        return jsonify(
            {
                "error": "No image uploaded.",
                "usage": (
                    "Send an image using form-data "
                    "with key 'image'."
                ),
            }
        ), 400

    try:

        # ----------------------------------------------------
        # Load model only when prediction is requested
        # ----------------------------------------------------

        load_model()

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image_file = request.files["image"]

        image = Image.open(
            image_file
        ).convert("RGB")

        # ----------------------------------------------------
        # Preprocess image
        # ----------------------------------------------------

        inputs = processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

        # ----------------------------------------------------
        # Model inference
        # ----------------------------------------------------

        with torch.no_grad():

            outputs = model(
                **inputs
            )

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=-1,
            ).item()

            confidence = probabilities[
                0,
                predicted_class,
            ].item()

        # ----------------------------------------------------
        # Get class label
        # ----------------------------------------------------

        label = model.config.id2label.get(
            predicted_class,
            f"LABEL_{predicted_class}",
        )

        # ----------------------------------------------------
        # Update Prometheus metrics
        # ----------------------------------------------------

        prediction_success.inc()

        prediction_confidence.set(
            float(confidence)
        )

        prediction_class_id.set(
            float(predicted_class)
        )

        prediction_latency.observe(
            time.time() - start_time
        )

        # ----------------------------------------------------
        # Response
        # ----------------------------------------------------

        return jsonify(
            {
                "model": "Food-101 trained DeiT-Tiny",
                "prediction": {
                    "class_id": predicted_class,
                    "confidence": round(
                        float(confidence),
                        4,
                    ),
                    "label": label,
                },
            }
        )

    except Exception as exc:

        prediction_errors.inc()

        prediction_latency.observe(
            time.time() - start_time
        )

        print(
            "Prediction error:",
            str(exc),
        )

        return jsonify(
            {
                "error": str(exc),
            }
        ), 500


# ============================================================
# RUN APPLICATION
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000,
        )
    )

    print("=" * 60)
    print("FOOD-101 INFERENCE API")
    print("=" * 60)
    print("Starting Flask server...")
    print("Port:", port)
    print("Model will be loaded on first prediction request.")
    print("=" * 60)

    app.run(
        host="0.0.0.0",
        port=port,
    )
