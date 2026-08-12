import mlflow
import mlflow.pytorch
import os
import time

import torch
from torch.utils.data import DataLoader

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
    MODEL_DIR,
)

from src.data import (
    load_food101,
    create_train_validation_split,
)


def prepare_batch(batch, processor):

    images = [
        item["image"].convert("RGB")
        for item in batch
    ]

    labels = torch.tensor(
        [
            item["label"]
            for item in batch
        ],
        dtype=torch.long
    )

    inputs = processor(
        images=images,
        return_tensors="pt"
    )

    inputs["labels"] = labels

    return inputs


def train_model():

    print("=" * 60)
    print("FOOD-101 BASELINE TRAINING")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    print("\nLoading Food-101 dataset...")

    dataset = load_food101()

    train_dataset, validation_dataset = (
        create_train_validation_split(dataset)
    )

    print(
        f"Training samples: {len(train_dataset)}"
    )

    print(
        f"Validation samples: {len(validation_dataset)}"
    )

    # --------------------------------------------------------
    # 2. Load processor
    # --------------------------------------------------------

    print("\nLoading image processor...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_NAME
    )

    # --------------------------------------------------------
    # 3. Load model
    # --------------------------------------------------------

    print("\nLoading model...")

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    # --------------------------------------------------------
    # 4. Device
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
    # 5. DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch:
            prepare_batch(
                batch,
                processor
            ),
    )

    # --------------------------------------------------------
    # 6. Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE
    )

    # --------------------------------------------------------
    # 7. Training
    # --------------------------------------------------------

    model.train()

    start_time = time.time()

    for epoch in range(EPOCHS):

        epoch_loss = 0.0

        print(
            f"\nEpoch {epoch + 1}/{EPOCHS}"
        )

        for batch_number, batch in enumerate(
            train_loader,
            start=1
        ):

            # Move tensors to CPU/GPU
            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            # Forward pass
            outputs = model(**batch)

            loss = outputs.loss

            # Backpropagation
            optimizer.zero_grad()

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

            # Progress
            if (
                batch_number == 1
                or batch_number % 10 == 0
                or batch_number == len(train_loader)
            ):

                print(
                    f"Batch "
                    f"{batch_number}/"
                    f"{len(train_loader)} "
                    f"| Loss: "
                    f"{loss.item():.4f}"
                )

        average_loss = (
            epoch_loss / len(train_loader)
        )

        print(
            f"\nEpoch {epoch + 1} "
            f"Average Loss: "
            f"{average_loss:.4f}"
        )

    # --------------------------------------------------------
    # 8. Training time
    # --------------------------------------------------------

    training_time = (
        time.time() - start_time
    )

    # --------------------------------------------------------
    # 9. Save model
    # --------------------------------------------------------

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    model.save_pretrained(
        MODEL_DIR
    )

    processor.save_pretrained(
        MODEL_DIR
    )

    # --------------------------------------------------------
    # 10. Summary
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("TRAINING COMPLETED")
    print("=" * 60)

    print(
        f"Training time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Model saved to: "
        f"{MODEL_DIR}"
    )

    print(
        f"Final training loss: "
        f"{average_loss:.4f}"
    )


if __name__ == "__main__":

    train_model()
