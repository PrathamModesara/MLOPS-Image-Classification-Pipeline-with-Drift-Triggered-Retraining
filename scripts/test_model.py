import torch

from transformers import (
    AutoImageProcessor,
    AutoModelForImageClassification,
)

from datasets import load_dataset


MODEL_NAME = "facebook/deit-tiny-patch16-224"


print("Loading Food-101 sample...")

dataset = load_dataset(
    "ethz/food101",
    split="train[:1]"
)

image = dataset[0]["image"].convert("RGB")

print("Image loaded successfully.")
print("Image size:", image.size)


print("\nLoading image processor...")

processor = AutoImageProcessor.from_pretrained(
    MODEL_NAME
)

print("Processor loaded.")


print("\nLoading pretrained model...")

model = AutoModelForImageClassification.from_pretrained(
    MODEL_NAME
)

model.eval()

print("Model loaded.")


print("\nPreparing image...")

inputs = processor(
    images=image,
    return_tensors="pt"
)


print("Running prediction...")

with torch.no_grad():

    outputs = model(**inputs)

    probabilities = torch.softmax(
        outputs.logits,
        dim=-1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=-1
    ).item()

    confidence = probabilities[
        0,
        predicted_class
    ].item()


print("\nPrediction complete.")

print("Predicted class ID:", predicted_class)

print(
    "Confidence:",
    round(confidence, 4)
)

print("\nModel test successful!")
