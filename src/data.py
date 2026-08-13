from datasets import load_dataset

from src.config import (
    DATASET_NAME,
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_IMAGES_PER_CLASS = 50
VALIDATION_IMAGES_PER_CLASS = 10


# ============================================================
# CREATE BALANCED SUBSET
# ============================================================

def create_balanced_subset(
    dataset,
    images_per_class,
):
    """
    Create a balanced subset without decoding all images.

    Example:
        101 classes x 50 images = 5050 images
    """

    labels = dataset["label"]

    class_indices = {}

    for index, label in enumerate(labels):

        if label not in class_indices:
            class_indices[label] = []

        if len(class_indices[label]) < images_per_class:
            class_indices[label].append(index)

    selected_indices = []

    for label in sorted(class_indices.keys()):

        selected_indices.extend(
            class_indices[label]
        )

    return dataset.select(
        selected_indices
    )


# ============================================================
# LOAD FOOD-101
# ============================================================

def load_food101():

    print("\nLoading Food-101 dataset...")

    # --------------------------------------------------------
    # Training metadata
    # --------------------------------------------------------

    print("Loading full training metadata...")

    full_train = load_dataset(
        DATASET_NAME,
        split="train",
    )

    # --------------------------------------------------------
    # Validation metadata
    # --------------------------------------------------------

    print("Loading full validation metadata...")

    full_validation = load_dataset(
        DATASET_NAME,
        split="validation",
    )

    # --------------------------------------------------------
    # Balanced training subset
    # --------------------------------------------------------

    print(
        "\nCreating balanced training subset..."
    )

    train_dataset = create_balanced_subset(
        full_train,
        TRAIN_IMAGES_PER_CLASS,
    )

    # --------------------------------------------------------
    # Balanced validation subset
    # --------------------------------------------------------

    print(
        "Creating balanced validation subset..."
    )

    validation_dataset = create_balanced_subset(
        full_validation,
        VALIDATION_IMAGES_PER_CLASS,
    )

    print(
        f"\nTraining samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    print(
        f"Training classes: "
        f"{len(set(train_dataset['label']))}"
    )

    print(
        f"Validation classes: "
        f"{len(set(validation_dataset['label']))}"
    )

    return {
        "train": train_dataset,
        "validation": validation_dataset,
    }


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

def create_train_validation_split(
    dataset,
):

    train_dataset = dataset["train"]

    validation_dataset = dataset["validation"]

    return (
        train_dataset,
        validation_dataset,
    )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    dataset = load_food101()

    train_dataset, validation_dataset = (
        create_train_validation_split(
            dataset
        )
    )

    print("\n" + "=" * 60)
    print("FOOD-101 BALANCED DEVELOPMENT DATASET")
    print("=" * 60)

    print(
        "Training samples:",
        len(train_dataset),
    )

    print(
        "Validation samples:",
        len(validation_dataset),
    )

    print(
        "Training columns:",
        train_dataset.column_names,
    )

    print(
        "Validation columns:",
        validation_dataset.column_names,
    )

    print(
        "Training classes:",
        len(
            set(
                train_dataset["label"]
            )
        ),
    )

    print(
        "Validation classes:",
        len(
            set(
                validation_dataset["label"]
            )
        ),
    )

    print("=" * 60)
