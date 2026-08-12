import importlib
import os
import sys

print("=" * 70)
print("FOOD-101 MLOps PROJECT HEALTH CHECK")
print("=" * 70)

print(f"\nPython version: {sys.version}")
print(f"Python executable: {sys.executable}")

# ---------------------------------------------------------
# 1. Check required packages
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("1. CHECKING REQUIRED LIBRARIES")
print("=" * 70)

packages = [
    "torch",
    "transformers",
    "datasets",
    "PIL",
    "numpy",
    "pandas",
    "sklearn",
    "mlflow",
    "optuna",
    "zenml",
    "sqlalchemy",
    "sqlalchemy_utils",
    "sqlmodel",
    "pydantic",
]

for package in packages:
    try:
        module = importlib.import_module(package)

        version = getattr(module, "__version__", "version not available")

        print(f"[OK] {package:20} {version}")

    except Exception as e:
        print(f"[FAIL] {package:20} -> {e}")


# ---------------------------------------------------------
# 2. Check project directories
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("2. CHECKING PROJECT DIRECTORIES")
print("=" * 70)

directories = [
    "src",
    "models",
    "data",
    "artifacts",
    "mlartifacts",
    "monitoring",
    "pipelines",
    "prometheus",
    "grafana",
    "api",
    "tests",
    ".github",
]

for directory in directories:

    if os.path.isdir(directory):
        print(f"[OK]   {directory}/")
    else:
        print(f"[FAIL] {directory}/")


# ---------------------------------------------------------
# 3. Check important files
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("3. CHECKING IMPORTANT FILES")
print("=" * 70)

files = [
    "requirements.txt",
    "mlflow.db",

    "src/__init__.py",
    "src/config.py",
    "src/data.py",
    "src/model.py",
    "src/train.py",
    "src/evaluate.py",
    "src/optuna_tuning.py",
    "src/zenml_pipeline.py",
]

for file in files:

    if os.path.isfile(file):
        print(f"[OK]   {file}")
    else:
        print(f"[FAIL] {file}")


# ---------------------------------------------------------
# 4. Check Python modules
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("4. CHECKING PYTHON MODULE IMPORTS")
print("=" * 70)

modules = [
    "src.config",
    "src.data",
    "src.model",
    "src.train",
    "src.evaluate",
    "src.optuna_tuning",
    "src.zenml_pipeline",
]

for module_name in modules:

    try:
        importlib.import_module(module_name)

        print(f"[OK]   {module_name}")

    except Exception as e:

        print(f"[FAIL] {module_name}")
        print(f"       {type(e).__name__}: {e}")


# ---------------------------------------------------------
# 5. Check model directory
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("5. CHECKING TRAINED MODEL")
print("=" * 70)

model_path = "models/food101_model"

if os.path.isdir(model_path):

    print(f"[OK] Model directory exists: {model_path}")

    model_files = os.listdir(model_path)

    if model_files:

        for file in model_files:
            print(f"     - {file}")

    else:

        print("[WARNING] Model directory is empty")

else:

    print("[FAIL] Model directory does not exist")


# ---------------------------------------------------------
# 6. Check MLflow database
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("6. CHECKING MLFLOW")
print("=" * 70)

if os.path.isfile("mlflow.db"):

    print("[OK] mlflow.db exists")

else:

    print("[FAIL] mlflow.db not found")


try:

    import mlflow

    mlflow.set_tracking_uri(
        os.getenv(
            "MLFLOW_TRACKING_URI",
            "http://127.0.0.1:5000"
        )
    )

    print(f"MLflow tracking URI: {mlflow.get_tracking_uri()}")

    experiments = mlflow.search_experiments()

    print(f"[OK] MLflow connection successful")
    print(f"     Experiments found: {len(experiments)}")

    for experiment in experiments:

        print(
            f"     - {experiment.name} "
            f"(ID: {experiment.experiment_id})"
        )

except Exception as e:

    print("[FAIL] MLflow connection")
    print(f"       {type(e).__name__}: {e}")


# ---------------------------------------------------------
# 7. Check ZenML
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("7. CHECKING ZENML")
print("=" * 70)

try:

    import zenml

    print(f"[OK] ZenML installed")
    print(f"     Version: {zenml.__version__}")

except Exception as e:

    print("[FAIL] ZenML")
    print(f"       {type(e).__name__}: {e}")


# ---------------------------------------------------------
# 8. Check PyTorch
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("8. CHECKING PYTORCH")
print("=" * 70)

try:

    import torch

    print(f"[OK] PyTorch version: {torch.__version__}")

    print(
        f"CUDA available: {torch.cuda.is_available()}"
    )

    if torch.cuda.is_available():

        print(
            f"GPU: {torch.cuda.get_device_name(0)}"
        )

    else:

        print("Device: CPU")

except Exception as e:

    print("[FAIL] PyTorch")
    print(f"       {type(e).__name__}: {e}")


# ---------------------------------------------------------
# FINAL
# ---------------------------------------------------------

print("\n" + "=" * 70)
print("PROJECT HEALTH CHECK COMPLETED")
print("=" * 70)

print("""
Next steps:

[1] Fix any [FAIL] library
[2] Fix any [FAIL] project file
[3] Fix any [FAIL] module
[4] Start MLflow server
[5] Run data
[6] Run training
[7] Run evaluation
[8] Run Optuna
[9] Continue with ZenML
""")

print("=" * 70)
