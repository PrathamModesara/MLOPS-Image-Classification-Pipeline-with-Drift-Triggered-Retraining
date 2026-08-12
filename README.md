# Food-101 Image Classification MLOps

## Project Overview

This project implements an MLOps pipeline for image classification using the Food-101 dataset.

The objective is to detect input drift and automatically trigger model retraining.

## Technologies

- Python
- PyTorch
- Hugging Face Datasets
- Hugging Face Transformers
- DeiT-Tiny
- MLflow
- Optuna
- ZenML
- Prometheus
- Grafana
- Docker
- GitHub Actions
- Render

## Dataset

Food-101 contains 101 food categories.

For development, a balanced subset is used:

- Training: 505 images
- Validation: 101 images
- Classes: 101

## Current Pipeline

```text
Food-101
   ↓
Data Preparation
   ↓
DeiT-Tiny Training
   ↓
Evaluation
   ↓
Optuna Hyperparameter Tuning
   ↓
MLflow Experiment Tracking
   ↓
ZenML Pipeline
   ↓
Drift Detection
   ↓
Retraining
