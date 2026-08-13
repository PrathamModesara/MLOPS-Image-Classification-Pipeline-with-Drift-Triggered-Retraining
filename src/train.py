import os
import time

import torch

from torch.utils.data import DataLoader

from torchvision import transforms

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

from src.config import (
    MODEL_NAME,
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS,
    LEARNING_RATE,
    AUGMENTATION_STRENGTH,
    MODEL_DIR,
)

from src.data import (
    load_food101,
    create_train_validation_split,
)


# ============================================================
# MEMORY SETTINGS
# ============================================================

GRADIENT_ACCUMULATION_STEPS = 4


# ============================================================
# LEARNING RATE STRATEGY
# ============================================================

# Optuna provides the base learning rate.
#
# We use a smaller learning rate for the pretrained
# backbone and a larger learning rate for the newly
# initialized classifier.

BACKBONE_LR = LEARNING_RATE * 0.1

CLASSIFIER_LR = LEARNING_RATE * 0.5


# ============================================================
# IMAGE AUGMENTATION
# ============================================================

def create_augmentation(
    strength,
):
    """
    Create training-time image augmentation.

    The strength value comes from the best Optuna trial.
    """

    # Convert Optuna's 0-0.5 range into reasonable
    # torchvision augmentation parameters.

    rotation = 30.0 * strength

    color_strength = 0.5 * strength

    return transforms.Compose(
        [
            transforms.RandomHorizontalFlip(
                p=min(
                    0.5,
                    strength * 5,
                )
            ),

            transforms.RandomRotation(
                degrees=rotation
            ),

            transforms.ColorJitter(
                brightness=color_strength,
                contrast=color_strength,
                saturation=color_strength,
                hue=min(
                    0.1,
                    strength * 0.2,
                ),
            ),
        ]
    )


# ============================================================
# PREPARE BATCH
# ============================================================

def prepare_batch(
    batch,
    processor,
    augmentation=None,
):

    images = [
        item["image"].convert("RGB")
        for item in batch
    ]

    # --------------------------------------------------------
    # Apply augmentation ONLY to training images
    # --------------------------------------------------------

    if augmentation is not None:

        images = [
            augmentation(image)
            for image in images
        ]

    labels = torch.tensor(
        [
            item["label"]
            for item in batch
        ],
        dtype=torch.long,
    )

    inputs = processor(
        images=images,
        return_tensors="pt",
    )

    inputs["labels"] = labels

    return inputs


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    print("=" * 60)
    print("FOOD-101 OPTUNA-OPTIMIZED TRAINING")
    print("=" * 60)

    print(
        f"\nOptuna learning rate: "
        f"{LEARNING_RATE}"
    )

    print(
        f"Optuna augmentation strength: "
        f"{AUGMENTATION_STRENGTH}"
    )

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    print("\nLoading Food-101 dataset...")

    dataset = load_food101()

    train_dataset, validation_dataset = (
        create_train_validation_split(
            dataset
        )
    )

    print(
        f"\nTraining samples: "
        f"{len(train_dataset)}"
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    # --------------------------------------------------------
    # 2. Image processor
    # --------------------------------------------------------

    print(
        "\nLoading image processor..."
    )

    processor = (
        AutoImageProcessor.from_pretrained(
            MODEL_NAME
        )
    )

    # --------------------------------------------------------
    # 3. Create augmentation
    # --------------------------------------------------------

    augmentation = create_augmentation(
        AUGMENTATION_STRENGTH
    )

    print(
        "\nTraining augmentation: ENABLED"
    )

    print(
        f"Augmentation strength: "
        f"{AUGMENTATION_STRENGTH}"
    )

    # --------------------------------------------------------
    # 4. Load model
    # --------------------------------------------------------

    print(
        "\nLoading pretrained model..."
    )

    model = (
        AutoModelForImageClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )
    )

    # --------------------------------------------------------
    # 5. Device
    # --------------------------------------------------------

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    print(
        f"\nTraining device: {device}"
    )

    model.to(device)

    # --------------------------------------------------------
    # 6. Gradient checkpointing
    # --------------------------------------------------------

    if hasattr(
        model,
        "gradient_checkpointing_enable",
    ):

        try:

            model.gradient_checkpointing_enable()

            print(
                "Gradient checkpointing: ENABLED"
            )

        except Exception:

            print(
                "Gradient checkpointing: "
                "NOT AVAILABLE"
            )

    # --------------------------------------------------------
    # 7. DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=False,
        collate_fn=lambda batch:
            prepare_batch(
                batch,
                processor,
                augmentation,
            ),
    )

    # --------------------------------------------------------
    # 8. Parameter groups
    # --------------------------------------------------------

    classifier_parameters = []

    backbone_parameters = []

    for name, parameter in (
        model.named_parameters()
    ):

        if not parameter.requires_grad:
            continue

        if (
            "classifier" in name
            or "head" in name
        ):

            classifier_parameters.append(
                parameter
            )

        else:

            backbone_parameters.append(
                parameter
            )

    optimizer = torch.optim.AdamW(
        [
            {
                "params":
                    backbone_parameters,
                "lr":
                    BACKBONE_LR,
            },
            {
                "params":
                    classifier_parameters,
                "lr":
                    CLASSIFIER_LR,
            },
        ],
        weight_decay=0.01,
    )

    print(
        f"\nBackbone learning rate: "
        f"{BACKBONE_LR}"
    )

    print(
        f"Classifier learning rate: "
        f"{CLASSIFIER_LR}"
    )

    print(
        f"Gradient accumulation: "
        f"{GRADIENT_ACCUMULATION_STEPS}"
    )

    # --------------------------------------------------------
    # 9. Training
    # --------------------------------------------------------

    model.train()

    start_time = time.time()

    best_loss = float("inf")

    best_state = None

    for epoch in range(EPOCHS):

        epoch_loss = 0.0

        optimizer.zero_grad(
            set_to_none=True
        )

        print(
            f"\nEpoch "
            f"{epoch + 1}/{EPOCHS}"
        )

        for (
            batch_number,
            batch
        ) in enumerate(
            train_loader,
            start=1,
        ):

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(
                **batch
            )

            loss = outputs.loss

            # ------------------------------------------------
            # Gradient accumulation
            # ------------------------------------------------

            scaled_loss = (
                loss
                / GRADIENT_ACCUMULATION_STEPS
            )

            scaled_loss.backward()

            if (
                batch_number
                % GRADIENT_ACCUMULATION_STEPS
                == 0
                or batch_number
                == len(train_loader)
            ):

                torch.nn.utils.clip_grad_norm_(
                    model.parameters(),
                    max_norm=1.0,
                )

                optimizer.step()

                optimizer.zero_grad(
                    set_to_none=True
                )

            epoch_loss += loss.item()

            # ------------------------------------------------
            # Progress
            # ------------------------------------------------

            if (
                batch_number == 1
                or batch_number % 100 == 0
                or batch_number
                == len(train_loader)
            ):

                print(
                    f"Batch "
                    f"{batch_number}/"
                    f"{len(train_loader)} "
                    f"| Loss: "
                    f"{loss.item():.4f}"
                )

        average_loss = (
            epoch_loss
            / len(train_loader)
        )

        print(
            f"\nEpoch "
            f"{epoch + 1} Average Loss: "
            f"{average_loss:.4f}"
        )

        # ----------------------------------------------------
        # Save best state
        # ----------------------------------------------------

        if average_loss < best_loss:

            best_loss = average_loss

            best_state = {
                key: value.detach().cpu().clone()
                for key, value
                in model.state_dict().items()
            }

            print(
                "New best training loss."
            )

    # --------------------------------------------------------
    # 10. Restore best model
    # --------------------------------------------------------

    if best_state is not None:

        model.load_state_dict(
            best_state
        )

        model.to(device)

    # --------------------------------------------------------
    # 11. Save model
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True,
    )

    print(
        "\nSaving trained model..."
    )

    model.save_pretrained(
        MODEL_DIR
    )

    processor.save_pretrained(
        MODEL_DIR
    )

    training_time = (
        time.time()
        - start_time
    )

    print(
        f"\nTraining time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Model saved to: "
        f"{MODEL_DIR}"
    )

    print(
        f"Final training loss: "
        f"{best_loss:.4f}"
    )

    return {
        "training_loss": best_loss,
        "training_time": training_time,
        "learning_rate": LEARNING_RATE,
        "augmentation_strength":
            AUGMENTATION_STRENGTH,
        "backbone_lr": BACKBONE_LR,
        "classifier_lr": CLASSIFIER_LR,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    train_model()
