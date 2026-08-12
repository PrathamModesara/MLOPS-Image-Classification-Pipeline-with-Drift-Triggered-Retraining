from datasets import load_dataset

from src.config import DATASET_NAME


TRAIN_IMAGES_PER_CLASS = 5
VALIDATION_IMAGES_PER_CLASS = 1

NUM_CLASSES = 101


def get_balanced_split(
    dataset,
    images_per_class
):
    """
    Select a fixed number of images
    from every Food-101 class.
    """

    selected_indices = []

    labels = dataset["label"]

    for class_id in range(NUM_CLASSES):

        class_indices = [
            i
            for i, label in enumerate(labels)
            if label == class_id
        ]

        if len(class_indices) < images_per_class:

            raise ValueError(
                f"Class {class_id} has only "
                f"{len(class_indices)} images."
            )

        selected_indices.extend(
            class_indices[:images_per_class]
        )

    return dataset.select(
        selected_indices
    )


def load_food101():

    print("\nLoading Food-101 dataset...")

    print(
        "Loading full training metadata..."
    )

    train_dataset = load_dataset(
        DATASET_NAME,
        split="train"
    )

    print(
        "Loading full validation metadata..."
    )

    validation_dataset = load_dataset(
        DATASET_NAME,
        split="validation"
    )

    print(
        "\nCreating balanced development split..."
    )

    train_dataset = get_balanced_split(
        train_dataset,
        TRAIN_IMAGES_PER_CLASS
    )

    validation_dataset = get_balanced_split(
        validation_dataset,
        VALIDATION_IMAGES_PER_CLASS
    )

    return {
        "train": train_dataset,
        "validation": validation_dataset,
    }


def create_train_validation_split(dataset):

    train_dataset = dataset["train"]

    validation_dataset = dataset["validation"]

    return (
        train_dataset,
        validation_dataset,
    )


if __name__ == "__main__":

    dataset = load_food101()

    train_dataset, validation_dataset = (
        create_train_validation_split(dataset)
    )

    print("\n" + "=" * 60)
    print("FOOD-101 BALANCED DEVELOPMENT DATASET")
    print("=" * 60)

    print(
        "Training samples:",
        len(train_dataset)
    )

    print(
        "Validation samples:",
        len(validation_dataset)
    )

    print(
        "Training columns:",
        train_dataset.column_names
    )

    print(
        "Validation columns:",
        validation_dataset.column_names
    )

    train_labels = sorted(
        set(train_dataset["label"])
    )

    validation_labels = sorted(
        set(validation_dataset["label"])
    )

    print(
        "Training classes:",
        len(train_labels)
    )

    print(
        "Validation classes:",
        len(validation_labels)
    )

    print(
        "Training labels:",
        train_labels
    )

    print(
        "Validation labels:",
        validation_labels
    )

    print("=" * 60)
