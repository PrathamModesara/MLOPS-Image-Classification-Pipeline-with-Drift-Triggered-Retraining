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
# DRIFT MONITORING SAMPLE SIZES
# ============================================================

# IMPORTANT:
# Training/evaluation still use:
#   Training   = 5050
#   Validation = 1010
#
# Drift monitoring uses a smaller representative sample:
#   Reference = 505
#   Current   = 101
#
# This makes CPU-based drift detection much faster.

DRIFT_REFERENCE_PER_CLASS = 5
DRIFT_CURRENT_PER_CLASS = 1


# ============================================================
# DEVICE
# ============================================================

def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


# ============================================================
# CREATE BALANCED DRIFT SAMPLE
# ============================================================

def create_balanced_drift_sample(
    dataset,
    samples_per_class,
):
    """
    Create a small balanced sample from a dataset.

    Food-101 contains 101 classes.

    For example:
        5 images/class  -> 505 images
        1 image/class   -> 101 images
    """

    labels = dataset["label"]

    selected_indices = []

    unique_labels = sorted(
        set(labels)
    )

    for label in unique_labels:

        class_indices = [
            index
            for index, current_label
            in enumerate(labels)
            if current_label == label
        ]

        # Deterministic selection.
        # This keeps the drift experiment reproducible.
        class_indices = sorted(
            class_indices
        )

        selected_indices.extend(
            class_indices[
                :samples_per_class
            ]
        )

    return dataset.select(
        selected_indices
    )


# ============================================================
# LOAD DRIFTED DATASET
# ============================================================

def load_drifted_dataset(
    directory="data/drifted_images",
):
    """
    Load the simulated drifted images.
    """

    if not os.path.exists(directory):

        raise FileNotFoundError(
            f"Drifted image directory not found: "
            f"{directory}"
        )

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

    # Use a small batch size on CPU.
    # BATCH_SIZE comes from config.py.
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=collate_images,
    )

    model.eval()

    with torch.no_grad():

        for batch_number, images in enumerate(
            loader,
            start=1,
        ):

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
                outputs.last_hidden_state[
                    :, 0, :
                ]
            )

            embeddings.extend(
                batch_embeddings
                .cpu()
                .numpy()
            )

            if (
                batch_number % 10 == 0
                or batch_number == len(loader)
            ):

                print(
                    f"Embedding batch "
                    f"{batch_number}/{len(loader)}"
                )

    return np.array(
        embeddings
    )


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
        reference_mean
        - current_mean
    )

    return float(
        drift_score
    )


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
# SAVE EMBEDDINGS
# ============================================================

def save_embeddings(
    reference_embeddings,
    current_embeddings,
):

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

    # --------------------------------------------------------
    # Create smaller balanced drift samples
    # --------------------------------------------------------

    reference_dataset = (
        create_balanced_drift_sample(
            train_dataset,
            DRIFT_REFERENCE_PER_CLASS,
        )
    )

    current_dataset = (
        create_balanced_drift_sample(
            validation_dataset,
            DRIFT_CURRENT_PER_CLASS,
        )
    )

    print(
        f"\nFull training images: "
        f"{len(train_dataset)}"
    )

    print(
        f"Full validation images: "
        f"{len(validation_dataset)}"
    )

    print(
        f"\nDrift reference images: "
        f"{len(reference_dataset)}"
    )

    print(
        f"Drift current images: "
        f"{len(current_dataset)}"
    )

    print(
        "\nCreating reference embeddings..."
    )

    reference_embeddings = create_embeddings(
        reference_dataset,
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
        current_dataset,
        model,
        processor,
        device,
    )

    print(
        "Current embedding shape:",
        current_embeddings.shape,
    )

    # Save reference/current embeddings
    save_embeddings(
        reference_embeddings,
        current_embeddings,
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

    # Keep exactly one drifted image
    # per Food-101 class.
    #
    # Your simulation creates 101 images,
    # so this should normally remain 101.

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
# MAIN DRIFT DETECTION
# ============================================================

def detect_drift(
    use_drifted_data=False,
):

    print("=" * 60)
    print(
        "FOOD-101 IMAGE DRIFT DETECTION"
    )
    print("=" * 60)

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = get_device()

    print(
        f"\nDevice: {device}"
    )

    # --------------------------------------------------------
    # Load processor
    # --------------------------------------------------------

    print(
        "\nLoading image processor..."
    )

    processor = (
        AutoImageProcessor.from_pretrained(
            MODEL_DIR
        )
    )

    # --------------------------------------------------------
    # Load model
    # --------------------------------------------------------

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

        dataset_type = (
            "simulated_drift"
        )

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
        drift_score
        > DRIFT_THRESHOLD
    )

    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print(
        "DRIFT RESULTS"
    )
    print("=" * 60)

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

    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    save_drift_result(
        drift_score,
        DRIFT_THRESHOLD,
        drift_detected,
        dataset_type,
    )

    print(
        f"\nResult saved to: "
        f"{DRIFT_RESULT}"
    )

    print(
        "\nDrift detection completed."
    )

    return {
        "dataset_type": dataset_type,
        "drift_score": drift_score,
        "threshold": DRIFT_THRESHOLD,
        "drift_detected": drift_detected,
    }


# ============================================================
# COMMAND LINE
# ============================================================

if __name__ == "__main__":

    import argparse

    parser = argparse.ArgumentParser(
        description=(
            "Food-101 image drift detection"
        )
    )

    parser.add_argument(
        "--drifted",
        action="store_true",
        help=(
            "Run drift detection "
            "against simulated drifted images"
        ),
    )

    args = parser.parse_args()

    detect_drift(
        use_drifted_data=args.drifted
    )
