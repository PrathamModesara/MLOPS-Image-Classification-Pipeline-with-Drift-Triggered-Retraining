import os
import json

import numpy as np
import torch

from torch.utils.data import DataLoader
from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

from src.config import (
    MODEL_DIR,
    BATCH_SIZE,
    DRIFT_REFERENCE,
    DRIFT_CURRENT,
    DRIFT_RESULT,
    DRIFT_THRESHOLD,
)

from src.data import (
    load_food101,
    create_train_validation_split,
)


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# CREATE IMAGE EMBEDDINGS
# ============================================================

def create_embeddings(dataset, model, processor, device):

    embeddings = []

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch: [
            item["image"].convert("RGB")
            for item in batch
        ],
    )

    model.eval()

    with torch.no_grad():

        for images in loader:

            inputs = processor(
                images=images,
                return_tensors="pt",
            )

            inputs = {
                key: value.to(device)
                for key, value in inputs.items()
            }

            # ------------------------------------------------
            # Get ViT hidden representation
            # ------------------------------------------------

            outputs = model.vit(**inputs)

            # CLS token embedding
            batch_embeddings = (
                outputs.last_hidden_state[:, 0, :]
            )

            embeddings.extend(
                batch_embeddings.cpu().numpy()
            )

    return np.array(embeddings)


# ============================================================
# CALCULATE DRIFT SCORE
# ============================================================

def calculate_drift_score(
    reference_embeddings,
    current_embeddings,
):

    reference_mean = np.mean(
        reference_embeddings,
        axis=0,
    )

    current_mean = np.mean(
        current_embeddings,
        axis=0,
    )

    distance = np.linalg.norm(
        reference_mean - current_mean
    )

    return float(distance)


# ============================================================
# SAVE DRIFT RESULT
# ============================================================

def save_drift_result(
    drift_score,
    threshold,
    drift_detected,
):

    os.makedirs(
        os.path.dirname(DRIFT_RESULT),
        exist_ok=True,
    )

    result = {
        "drift_score": drift_score,
        "threshold": threshold,
        "drift_detected": drift_detected,
    }

    with open(
        DRIFT_RESULT,
        "w",
    ) as file:

        json.dump(
            result,
            file,
            indent=4,
        )


# ============================================================
# MAIN DRIFT DETECTION
# ============================================================

def detect_drift():

    print("=" * 60)
    print("FOOD-101 IMAGE DRIFT DETECTION")
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "\nLoading Food-101 dataset..."
    )

    dataset = load_food101()

    train_dataset, validation_dataset = (
        create_train_validation_split(dataset)
    )

    print(
        f"Reference images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Current images: "
        f"{len(validation_dataset)}"
    )

    # --------------------------------------------------------
    # Load processor
    # --------------------------------------------------------

    print(
        "\nLoading image processor..."
    )

    processor = AutoImageProcessor.from_pretrained(
        MODEL_DIR
    )

    # --------------------------------------------------------
    # Load trained model
    # --------------------------------------------------------

    print(
        "\nLoading trained model..."
    )

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_DIR
    )

    model.to(device)

    # --------------------------------------------------------
    # Reference embeddings
    # --------------------------------------------------------

    print(
        "\nCreating reference embeddings..."
    )

    reference_embeddings = create_embeddings(
        train_dataset,
        model,
        processor,
        device,
    )

    print(
        "Reference embedding shape:",
        reference_embeddings.shape,
    )

    # --------------------------------------------------------
    # Current embeddings
    # --------------------------------------------------------

    print(
        "\nCreating current embeddings..."
    )

    current_embeddings = create_embeddings(
        validation_dataset,
        model,
        processor,
        device,
    )

    print(
        "Current embedding shape:",
        current_embeddings.shape,
    )

    # --------------------------------------------------------
    # Save embeddings
    # --------------------------------------------------------

    os.makedirs(
        "artifacts",
        exist_ok=True,
    )

    np.save(
        DRIFT_REFERENCE,
        reference_embeddings,
    )

    np.save(
        DRIFT_CURRENT,
        current_embeddings,
    )

    # --------------------------------------------------------
    # Calculate drift
    # --------------------------------------------------------

    print(
        "\nCalculating drift score..."
    )

    drift_score = calculate_drift_score(
        reference_embeddings,
        current_embeddings,
    )

    drift_detected = (
        drift_score > DRIFT_THRESHOLD
    )

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    save_drift_result(
        drift_score,
        DRIFT_THRESHOLD,
        drift_detected,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("DRIFT RESULTS")
    print("=" * 60)

    print(
        f"\nDrift Score : "
        f"{drift_score:.4f}"
    )

    print(
        f"Threshold   : "
        f"{DRIFT_THRESHOLD:.4f}"
    )

    print(
        f"Drift       : "
        f"{drift_detected}"
    )

    if drift_detected:

        print(
            "\n⚠️ DRIFT DETECTED"
        )

    else:

        print(
            "\n✅ NO SIGNIFICANT DRIFT"
        )

    print(
        f"\nResult saved to: "
        f"{DRIFT_RESULT}"
    )

    print(
        "\nDrift detection completed."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    detect_drift()
