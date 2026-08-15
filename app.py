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

MODEL_DIR = os.path.join(
    os.path.dirname(__file__),
    "models",
    "food101_model",
)

app = Flask(__name__)

# Use CPU because Render Free does not provide GPU.
DEVICE = torch.device("cpu")

# Reduce CPU thread usage and memory overhead.
torch.set_num_threads(1)


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
# LOAD MODEL
# ============================================================

print("=" * 60)
print("FOOD-101 INFERENCE API")
print("=" * 60)

print("Model directory:", MODEL_DIR)
print("Device:", DEVICE)

print("\nLoading image processor...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_DIR,
)

print("Image processor loaded.")

print("\nLoading trained Food-101 model...")

model = AutoModelForImageClassification.from_pretrained(
    MODEL_DIR,
    low_cpu_mem_usage=True,
    use_safetensors=True,
)

model.to(DEVICE)
model.eval()

print("Trained Food-101 model loaded.")
print("Number of classes:", model.config.num_labels)


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route("/health", methods=["GET"])
def health():

    return jsonify(
        {
            "status": "ok",
            "service": "Food-101 Inference API",
            "model_loaded": True,
            "num_classes": model.config.num_labels,
        }
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

@app.route("/model-info", methods=["GET"])
def model_info():

    return jsonify(
        {
            "model": "Food-101 trained DeiT-Tiny",
            "num_classes": model.config.num_labels,
            "device": str(DEVICE),
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
# PREDICTION
# ============================================================

@app.route("/predict", methods=["POST"])
def predict():

    start_time = time.time()

    prediction_requests.inc()

    if "image" not in request.files:

        prediction_errors.inc()

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

        image_file = request.files["image"]

        image = Image.open(
            image_file
        ).convert("RGB")

        inputs = processor(
            images=image,
            return_tensors="pt",
        )

        inputs = {
            key: value.to(DEVICE)
            for key, value in inputs.items()
        }

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
        # Get model label if available
        # ----------------------------------------------------

        label = model.config.id2label.get(
            predicted_class,
            f"class_{predicted_class}",
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

        return jsonify(
            {
                "prediction": {
                    "class_id": predicted_class,
                    "label": label,
                    "confidence": round(
                        float(confidence),
                        4,
                    ),
                },
                "model": "Food-101 trained DeiT-Tiny",
            }
        )

    except Exception as exc:

        prediction_errors.inc()

        prediction_latency.observe(
            time.time() - start_time
        )

        return jsonify(
            {
                "error": str(exc),
            }
        ), 500


# ============================================================
# ROOT
# ============================================================

@app.route("/", methods=["GET"])
def home():

    return jsonify(
        {
            "message": "Food-101 Image Classification API",
            "endpoints": {
                "health": "/health",
                "model_info": "/model-info",
                "prediction": "POST /predict",
                "metrics": "/metrics",
            },
        }
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    port = int(
        os.environ.get(
            "PORT",
            8000,
        )
    )

    app.run(
        host="0.0.0.0",
        port=port,
    )
