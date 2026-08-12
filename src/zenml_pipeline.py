from zenml import pipeline, step


@step
def data_step():
    from src.data import load_food101, create_train_validation_split

    print("=" * 60)
    print("ZENML DATA STEP")
    print("=" * 60)

    dataset = load_food101()

    train_dataset, validation_dataset = create_train_validation_split(
        dataset
    )

    print(f"Training samples: {len(train_dataset)}")
    print(f"Validation samples: {len(validation_dataset)}")
    print(f"Training classes: {len(set(train_dataset['label']))}")
    print(f"Validation classes: {len(set(validation_dataset['label']))}")

    return len(train_dataset), len(validation_dataset)


@step
def training_step():
    from src.train import train_model

    print("=" * 60)
    print("ZENML TRAINING STEP")
    print("=" * 60)

    train_model()

    return "Training completed"


@step
def evaluation_step():
    from src.evaluate import evaluate_model

    print("=" * 60)
    print("ZENML EVALUATION STEP")
    print("=" * 60)

    metrics = evaluate_model()

    print("\nEvaluation metrics:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")

    return metrics


@pipeline
def food101_pipeline():
    data_step()
    training_step()
    evaluation_step()


if __name__ == "__main__":
    print("=" * 60)
    print("FOOD-101 ZENML PIPELINE")
    print("=" * 60)

    food101_pipeline()

    print("\nZenML pipeline execution completed.")
