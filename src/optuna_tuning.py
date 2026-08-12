import os
import shutil

import optuna
import mlflow

import torch
from torch.utils.data import DataLoader

from PIL import ImageEnhance

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

from sklearn.metrics import accuracy_score

from src.config import (
    MODEL_NAME,
    NUM_CLASSES,
    BATCH_SIZE,
    EPOCHS,
    MODEL_DIR,
    MLFLOW_TRACKING_URI,
    MLFLOW_EXPERIMENT,
)

from src.data import (
    load_food101,
    create_train_validation_split,
)


# ============================================================
# CONFIGURATION
# ============================================================

N_TRIALS = 3

OPTUNA_EPOCHS = 1

OPTUNA_MODEL_DIR = "models/optuna_trial_model"


# ============================================================
# IMAGE AUGMENTATION
# ============================================================

def augment_image(image, strength):

    image = image.convert("RGB")

    if strength <= 0:
        return image

    # Random brightness
    brightness_factor = 1.0 + (
        torch.rand(1).item() * 2 - 1
    ) * strength

    image = ImageEnhance.Brightness(
        image
    ).enhance(
        brightness_factor
    )

    # Random contrast
    contrast_factor = 1.0 + (
        torch.rand(1).item() * 2 - 1
    ) * strength

    image = ImageEnhance.Contrast(
        image
    ).enhance(
        contrast_factor
    )

    return image


# ============================================================
# BATCH PREPARATION
# ============================================================

def prepare_batch(
    batch,
    processor,
    augmentation_strength=0.0,
):

    images = []

    labels = []

    for item in batch:

        image = item["image"].convert("RGB")

        image = augment_image(
            image,
            augmentation_strength,
        )

        images.append(image)

        labels.append(
            item["label"]
        )

    labels = torch.tensor(
        labels,
        dtype=torch.long,
    )

    inputs = processor(
        images=images,
        return_tensors="pt",
    )

    inputs["labels"] = labels

    return inputs


# ============================================================
# EVALUATION
# ============================================================

def evaluate_trial(
    model,
    validation_loader,
    device,
):

    model.eval()

    total_loss = 0.0

    all_predictions = []

    all_labels = []

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

            total_loss += loss.item()

            all_predictions.extend(
                predictions.cpu().numpy()
            )

            all_labels.extend(
                batch["labels"].cpu().numpy()
            )

    validation_loss = (
        total_loss / len(validation_loader)
    )

    accuracy = accuracy_score(
        all_labels,
        all_predictions,
    )

    return (
        validation_loss,
        accuracy,
    )


# ============================================================
# OPTUNA OBJECTIVE
# ============================================================

def objective(trial):

    # --------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------

    learning_rate = trial.suggest_float(
        "learning_rate",
        1e-5,
        1e-3,
        log=True,
    )

    augmentation_strength = trial.suggest_float(
        "augmentation_strength",
        0.0,
        0.5,
    )

    print("\n" + "=" * 60)
    print(
        f"OPTUNA TRIAL {trial.number}"
    )
    print("=" * 60)

    print(
        f"Learning rate: "
        f"{learning_rate:.7f}"
    )

    print(
        f"Augmentation strength: "
        f"{augmentation_strength:.4f}"
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    dataset = load_food101()

    train_dataset, validation_dataset = (
        create_train_validation_split(
            dataset
        )
    )

    # --------------------------------------------------------
    # Processor
    # --------------------------------------------------------

    processor = (
        AutoImageProcessor.from_pretrained(
            MODEL_NAME
        )
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------

    model = (
        AutoModelForImageClassification
        .from_pretrained(
            MODEL_NAME,
            num_labels=NUM_CLASSES,
            ignore_mismatched_sizes=True,
        )
    )

    # --------------------------------------------------------
    # Device
    # --------------------------------------------------------

    device = (
        torch.device("cuda")
        if torch.cuda.is_available()
        else torch.device("cpu")
    )

    model.to(device)

    # --------------------------------------------------------
    # DataLoader
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=lambda batch:
            prepare_batch(
                batch,
                processor,
                augmentation_strength,
            ),
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=lambda batch:
            prepare_batch(
                batch,
                processor,
                0.0,
            ),
    )

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=learning_rate,
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    model.train()

    for epoch in range(
        OPTUNA_EPOCHS
    ):

        total_train_loss = 0.0

        print(
            f"\nEpoch "
            f"{epoch + 1}/"
            f"{OPTUNA_EPOCHS}"
        )

        for batch_number, batch in enumerate(
            train_loader,
            start=1,
        ):

            batch = {
                key: value.to(device)
                for key, value in batch.items()
            }

            optimizer.zero_grad()

            outputs = model(
                **batch
            )

            loss = outputs.loss

            loss.backward()

            optimizer.step()

            total_train_loss += (
                loss.item()
            )

            if (
                batch_number == 1
                or batch_number % 20 == 0
                or batch_number == len(train_loader)
            ):

                print(
                    f"Batch "
                    f"{batch_number}/"
                    f"{len(train_loader)} "
                    f"| Loss: "
                    f"{loss.item():.4f}"
                )

        average_train_loss = (
            total_train_loss
            / len(train_loader)
        )

        print(
            f"Average training loss: "
            f"{average_train_loss:.4f}"
        )

    # --------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------

    validation_loss, accuracy = (
        evaluate_trial(
            model,
            validation_loader,
            device,
        )
    )

    print(
        f"\nValidation Loss: "
        f"{validation_loss:.4f}"
    )

    print(
        f"Validation Accuracy: "
        f"{accuracy:.4f}"
    )

    # --------------------------------------------------------
    # Store Optuna values
    # --------------------------------------------------------

    trial.set_user_attr(
        "validation_loss",
        validation_loss,
    )

    trial.set_user_attr(
        "accuracy",
        accuracy,
    )

    # --------------------------------------------------------
    # MLflow nested run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name=f"optuna-trial-{trial.number}",
        nested=True,
    ):

        mlflow.log_param(
            "trial_number",
            trial.number,
        )

        mlflow.log_param(
            "learning_rate",
            learning_rate,
        )

        mlflow.log_param(
            "augmentation_strength",
            augmentation_strength,
        )

        mlflow.log_param(
            "epochs",
            OPTUNA_EPOCHS,
        )

        mlflow.log_param(
            "batch_size",
            BATCH_SIZE,
        )

        mlflow.log_metric(
            "validation_loss",
            validation_loss,
        )

        mlflow.log_metric(
            "accuracy",
            accuracy,
        )

        mlflow.set_tag(
            "stage",
            "optuna",
        )

        mlflow.set_tag(
            "framework",
            "huggingface-transformers",
        )

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    del model

    if torch.cuda.is_available():

        torch.cuda.empty_cache()

    return validation_loss


# ============================================================
# MAIN OPTUNA FUNCTION
# ============================================================

def run_optuna():

    print("=" * 60)
    print("FOOD-101 OPTUNA HYPERPARAMETER TUNING")
    print("=" * 60)

    # --------------------------------------------------------
    # MLflow
    # --------------------------------------------------------

    mlflow.set_tracking_uri(
        MLFLOW_TRACKING_URI
    )

    mlflow.set_experiment(
        MLFLOW_EXPERIMENT
    )

    # --------------------------------------------------------
    # Create Optuna study
    # --------------------------------------------------------

    study = optuna.create_study(
        direction="minimize",
        study_name="food101_deit_tiny",
    )

    # --------------------------------------------------------
    # Parent MLflow run
    # --------------------------------------------------------

    with mlflow.start_run(
        run_name="optuna-parent-run"
    ):

        mlflow.set_tag(
            "stage",
            "hyperparameter-tuning",
        )

        mlflow.set_tag(
            "framework",
            "optuna",
        )

        # ----------------------------------------------------
        # Optimize
        # ----------------------------------------------------

        study.optimize(
            objective,
            n_trials=N_TRIALS,
        )

        # ----------------------------------------------------
        # Best trial
        # ----------------------------------------------------

        best_trial = study.best_trial

        print("\n" + "=" * 60)
        print("OPTUNA RESULTS")
        print("=" * 60)

        print(
            "Best trial:",
            best_trial.number,
        )

        print(
            "Best validation loss:",
            best_trial.value,
        )

        print(
            "\nBest parameters:"
        )

        for name, value in (
            best_trial.params.items()
        ):

            print(
                f"{name}: {value}"
            )

            mlflow.log_param(
                f"best_{name}",
                value,
            )

        mlflow.log_metric(
            "best_validation_loss",
            best_trial.value,
        )

        mlflow.log_metric(
            "best_accuracy",
            best_trial.user_attrs.get(
                "accuracy",
                0.0,
            ),
        )

        print("=" * 60)


if __name__ == "__main__":

    run_optuna()
