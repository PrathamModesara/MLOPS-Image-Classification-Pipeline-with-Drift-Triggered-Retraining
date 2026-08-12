import torch

from torch.utils.data import DataLoader

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

from src.config import (
    MODEL_DIR,
    BATCH_SIZE,
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
        dtype=torch.long,
    )

    inputs = processor(
        images=images,
        return_tensors="pt",
    )

    inputs["labels"] = labels

    return inputs


def evaluate_model():

    print("=" * 60)
    print("FOOD-101 MODEL EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # 1. Load dataset
    # --------------------------------------------------------

    print("\nLoading dataset...")

    dataset = load_food101()

    _, validation_dataset = (
        create_train_validation_split(dataset)
    )

    print(
        f"Validation samples: "
        f"{len(validation_dataset)}"
    )

    # --------------------------------------------------------
    # 2. Load image processor
    # --------------------------------------------------------

    print("\nLoading processor...")

    processor = AutoImageProcessor.from_pretrained(
        MODEL_DIR
    )

    # --------------------------------------------------------
    # 3. Load trained model
    # --------------------------------------------------------

    print("\nLoading trained model...")

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_DIR
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
        f"Evaluation device: {device}"
    )

    model.to(device)

    model.eval()

    # --------------------------------------------------------
    # 5. DataLoader
    # --------------------------------------------------------

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch:
            prepare_batch(
                batch,
                processor,
            ),
    )

    # --------------------------------------------------------
    # 6. Prediction
    # --------------------------------------------------------

    all_predictions = []
    all_labels = []

    total_loss = 0.0

    print("\nRunning evaluation...")

    with torch.no_grad():

        for batch in validation_loader:

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            outputs = model(
                **batch
            )

            loss = outputs.loss

            predictions = torch.argmax(
                outputs.logits,
                dim=-1,
            )

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                batch["labels"].cpu().numpy()
            )

            total_loss += loss.item()

    # --------------------------------------------------------
    # 7. Calculate metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        average="weighted",
        zero_division=0,
    )

    validation_loss = (
        total_loss / len(validation_loader)
    )

    # --------------------------------------------------------
    # 8. Print results
    # --------------------------------------------------------

    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)

    print(
        f"\nValidation Loss : "
        f"{validation_loss:.4f}"
    )

    print(
        f"Accuracy        : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision       : "
        f"{precision:.4f}"
    )

    print(
        f"Recall          : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score        : "
        f"{f1:.4f}"
    )

    print("\nEvaluation completed.")

    # --------------------------------------------------------
    # 9. Return metrics for MLflow / ZenML
    # --------------------------------------------------------

    return {
        "validation_loss": validation_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
    }


if __name__ == "__main__":

    evaluate_model()
