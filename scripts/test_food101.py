from datasets import load_dataset


print("Loading Food-101...")


dataset = load_dataset(
    "ethz/food101",
    split="train[:10]"
)


print("\nDataset:")
print(dataset)


print("\nNumber of images:")
print(len(dataset))


print("\nColumns:")
print(dataset.column_names)


print("\nNumber of classes:")
print(
    len(dataset.features["label"].names)
)


print("\nFirst 10 classes:")
print(
    dataset.features["label"].names[:10]
)


sample = dataset[0]


print("\nFirst image:")
print(sample["image"])


print("\nFirst label ID:")
print(sample["label"])


print("\nFirst label name:")
print(
    dataset.features["label"].names[
        sample["label"]
    ]
)


print("\nFood-101 test successful!")
