import os
import json

import numpy as np
import torch

from PIL import Image
from torch.utils.data import DataLoader
from datasets import Dataset
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
# LOAD DRIFTED DATASET
# ============================================================

def load_drifted_dataset(
    directory="data/drifted_images",
):
    """
    Load the simulated drifted images.
    """

    image_files = sorted(
        [
            file
            for file in os.listdir(directory)
            if file.endswith(".jpg")
        ]
    )

    images = []

    for file in image_files:

        image_path = os.path.join(
            directory,
            file,
        )

        image = Image.open(
            image_path
        ).convert("RGB")

        images.append(image)

    dataset = Dataset.from_dict(
        {
            "image": images,
        }
    )

    return dataset


# ============================================================
# CREATE EMBEDDINGS
# ============================================================

def create_embeddings(
    dataset,
    model,
    processor,
    device,
):

    embeddings = []

    def collate_images(batch):

        return [
            item["image"].convert("RGB")
            for item in batch
        ]

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_images,
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

            outputs = model.vit(
                **inputs
            )

            batch_embeddings = (
                outputs.last_hidden_state[:, 0, :]
            )

            embeddings.extend(
                batch_embeddings
                .cpu()
                .numpy()
            )

    return np.array(embeddings)


# ============================================================
# CALCULATE DRIFT
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

    drift_score = np.linalg.norm(
        reference_mean - current_mean
    )

    return float(drift_score)


# ============================================================
# SAVE RESULT
# ============================================================

def save_drift_result(
    drift_score,
    threshold,
    drift_detected,
    dataset_type,
):

    os.makedirs(
        "artifacts",
        exist_ok=True,
    )

    result = {
        "dataset_type": dataset_type,
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
# NORMAL DRIFT CHECK
# ============================================================

def run_normal_drift_check(
    model,
    processor,
    device,
):

    print(
        "\nLoading normal validation data..."
    )

    dataset = load_food101()

    train_dataset, validation_dataset = (
        create_train_validation_split(
            dataset
        )
    )

    print(
        f"Reference images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Current images: "
        f"{len(validation_dataset)}"
    )

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

    print(
        "\nCreating normal current embeddings..."
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

    return (
        reference_embeddings,
        current_embeddings,
    )


# ============================================================
# DRIFTED DATA CHECK
# ============================================================

def run_drifted_check(
    model,
    processor,
    device,
):

    print(
        "\nLoading drifted images..."
    )

    drifted_dataset = (
        load_drifted_dataset()
    )

    print(
        f"Drifted images: "
        f"{len(drifted_dataset)}"
    )

    # --------------------------------------------------------
    # Load existing reference embeddings
    # --------------------------------------------------------

    if not os.path.exists(
        DRIFT_REFERENCE
    ):

        raise FileNotFoundError(
            "Reference embeddings not found. "
            "Run the normal drift check first."
        )

    reference_embeddings = np.load(
        DRIFT_REFERENCE
    )

    print(
        "\nReference embedding shape:",
        reference_embeddings.shape,
    )

    # --------------------------------------------------------
    # Create drifted embeddings
    # --------------------------------------------------------

    print(
        "\nCreating drifted embeddings..."
    )

    current_embeddings = create_embeddings(
        drifted_dataset,
        model,
        processor,
        device,
    )

    print(
        "Drifted embedding shape:",
        current_embeddings.shape,
    )

    return (
        reference_embeddings,
        current_embeddings,
    )


# ============================================================
# MAIN
# ============================================================

def detect_drift(
    use_drifted_data=False,
):

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
    # Load model
    # --------------------------------------------------------

    print(
        "\nLoading image processor..."
    )

    processor = (
        AutoImageProcessor.from_pretrained(
            MODEL_DIR
        )
    )

    print(
        "\nLoading trained model..."
    )

    model = (
        AutoModelForImageClassification
        .from_pretrained(
            MODEL_DIR
        )
    )

    model.to(device)

    # --------------------------------------------------------
    # Select dataset
    # --------------------------------------------------------

    if use_drifted_data:

        print(
            "\nMODE: SIMULATED DRIFT"
        )

        (
            reference_embeddings,
            current_embeddings,
        ) = run_drifted_check(
            model,
            processor,
            device,
        )

        dataset_type = "simulated_drift"

    else:

        print(
            "\nMODE: NORMAL DATA"
        )

        (
            reference_embeddings,
            current_embeddings,
        ) = run_normal_drift_check(
            model,
            processor,
            device,
        )

        dataset_type = "normal"

        # Save reference embeddings
        np.save(
            DRIFT_REFERENCE,
            reference_embeddings,
        )

    # --------------------------------------------------------
    # Save current embeddings
    # --------------------------------------------------------

    np.save(
        DRIFT_CURRENT,
        current_embeddings,
    )

    # --------------------------------------------------------
    # Calculate score
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
        dataset_type,
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print(
        "\n" + "=" * 60
    )

    print(
        "DRIFT RESULTS"
    )

    print(
        "=" * 60
    )

    print(
        f"\nDataset Type : "
        f"{dataset_type}"
    )

    print(
        f"Drift Score  : "
        f"{drift_score:.4f}"
    )

    print(
        f"Threshold    : "
        f"{DRIFT_THRESHOLD:.4f}"
    )

    print(
        f"Drift        : "
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

    import sys

    if "--drifted" in sys.argv:

        detect_drift(
            use_drifted_data=True
        )

    else:

        detect_drift(
            use_drifted_data=False
        )
