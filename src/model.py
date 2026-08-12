import torch

from transformers import (
    AutoModelForImageClassification,
)

from src.config import MODEL_NAME, NUM_CLASSES


def get_device():

    if torch.cuda.is_available():
        return torch.device("cuda")

    return torch.device("cpu")


def create_model():

    model = AutoModelForImageClassification.from_pretrained(
        MODEL_NAME,
        num_labels=NUM_CLASSES,
        ignore_mismatched_sizes=True,
    )

    return model


if __name__ == "__main__":

    device = get_device()

    print("Device:", device)

    model = create_model()

    print("\nModel loaded successfully.")

    print(
        "Number of labels:",
        model.config.num_labels
    )
