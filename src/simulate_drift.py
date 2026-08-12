import os

from PIL import ImageEnhance, ImageFilter
from datasets import Dataset

from src.data import (
    load_food101,
    create_train_validation_split,
)


def apply_drift(image, strength=1.8):
    """
    Simulate visual drift by changing
    brightness, contrast and sharpness.
    """

    image = image.convert("RGB")

    # --------------------------------------------------------
    # Brightness drift
    # --------------------------------------------------------

    image = ImageEnhance.Brightness(
        image
    ).enhance(strength)

    # --------------------------------------------------------
    # Contrast drift
    # --------------------------------------------------------

    image = ImageEnhance.Contrast(
        image
    ).enhance(1.5)

    # --------------------------------------------------------
    # Blur drift
    # --------------------------------------------------------

    image = image.filter(
        ImageFilter.GaussianBlur(
            radius=1.2
        )
    )

    return image


def create_drifted_dataset(
    dataset,
    output_dir="data/drifted_images",
    strength=1.8,
):
    """
    Create drifted copies of images.
    """

    os.makedirs(
        output_dir,
        exist_ok=True,
    )

    drifted_images = []
    labels = []

    print(
        f"\nCreating drifted images..."
    )

    for index, item in enumerate(dataset):

        image = item["image"]

        drifted_image = apply_drift(
            image,
            strength=strength,
        )

        image_path = os.path.join(
            output_dir,
            f"drifted_{index}.jpg",
        )

        drifted_image.save(
            image_path,
            quality=90,
        )

        drifted_images.append(
            drifted_image
        )

        labels.append(
            item["label"]
        )

        if (
            index == 0
            or (index + 1) % 20 == 0
            or index + 1 == len(dataset)
        ):
            print(
                f"Created "
                f"{index + 1}/"
                f"{len(dataset)}"
            )

    drifted_dataset = Dataset.from_dict(
        {
            "image": drifted_images,
            "label": labels,
        }
    )

    return drifted_dataset


def main():

    print("=" * 60)
    print("FOOD-101 DRIFT SIMULATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print(
        "\nLoading Food-101 dataset..."
    )

    dataset = load_food101()

    _train_dataset, validation_dataset = (
        create_train_validation_split(
            dataset
        )
    )

    print(
        f"Original validation images: "
        f"{len(validation_dataset)}"
    )

    # --------------------------------------------------------
    # Create drifted images
    # --------------------------------------------------------

    drifted_dataset = create_drifted_dataset(
        validation_dataset,
        strength=1.8,
    )

    print(
        "\n" + "=" * 60
    )

    print(
        "DRIFT SIMULATION COMPLETED"
    )

    print("=" * 60)

    print(
        f"\nDrifted images: "
        f"{len(drifted_dataset)}"
    )

    print(
        "Location: "
        "data/drifted_images/"
    )


if __name__ == "__main__":
    main()
