# Assignment 2 - Model Training & Automation with Azure 

> End-to-end MLOps pipeline for Amazon review sentiment classification using Azure Machine Learning, MLflow, Scikit-learn, and Azure DevOps CI/CD.

---

## Table of Contents

- [Overview](#overview)
- [Project Structure](#project-structure)
- [Part I – Data & Preprocessing](#part-i--data--preprocessing)
- [Part II – Model Training](#part-ii--model-training)
- [Part III – Azure ML Pipeline Architecture](#part-iii--azure-ml-pipeline-architecture)
- [Part IV – Model Deployment](#part-iv--model-deployment)
- [Part V – CI/CD with Azure DevOps](#part-v--cicd-with-azure-devops)
- [Part VI – Results & Evaluation](#part-vi--results--evaluation)
- [Part VII – Challenges & Solutions](#part-vii--challenges--solutions)
- [Part VIII – Versioning & Best Practices](#part-viii--versioning--best-practices)
- [Bonus: Known Limitation](#bonus-known-limitation)

---

## Overview

This project builds a complete end-to-end machine learning pipeline for sentiment classification of Amazon reviews using Azure Machine Learning. The goal is to classify text reviews into sentiment categories using a structured pipeline that includes data preprocessing, feature engineering, model training, evaluation, hyperparameter tuning, and deployment.

The system is designed as a production-style MLOps workflow using:

- **Azure ML Pipelines** for reproducible training
- **MLflow** for experiment tracking
- **Scikit-learn** for modeling
- **Azure DevOps (CI/CD)** for automation
- **Managed Endpoints** for real-time inference

Unlike traditional notebook-based workflows, this project emphasizes modularity, reproducibility, and deployment readiness.

---

## Project Structure
```
repo/
├── src/
│   ├── train.py              # Model training script
│   ├── score.py              # Inference/scoring script
│   ├── invoke_endpoint.py    # Endpoint testing client
│   └── sentiment.py          # Sentiment utilities
├── jobs/
│   ├── train_job.yml         # Training job definition
│   ├── sweep_job.yml         # Hyperparameter sweep job
│   └── deployment.yml        # Deployment configuration
├── env/
│   ├── conda.yml             # Training environment
│   └── inference_conda.yml   # Inference environment
├── azure-pipelines.yml       # CI/CD pipeline definition
└── README.md
```

---

## Part I – Data & Preprocessing

### Dataset Overview

The dataset consists of Amazon product reviews with text fields and associated sentiment labels. Each review contains:

- Review text
- Metadata (length, word count, etc.)
- Engineered NLP embeddings and statistical features
- Sentiment label (positive/negative or multi-class depending on configuration)

### Feature Engineering Pipeline

Raw text is transformed into ML-ready features using:

| Feature Type | Description |
|---|---|
| TF-IDF Vectorization | Captures word importance across documents |
| Sentence-BERT Embeddings | Captures semantic meaning |
| Statistical Features | Review length, word count, avg word length, punctuation frequency |
| Sentiment Lexicon Signals | VADER-based features |

These features are combined into a unified feature matrix used for model training.

### Train–Validation–Test Split

The dataset is split into three sets:

- **Training set** – model learning
- **Validation set** – hyperparameter tuning and sweep optimization
- **Test set** – final evaluation

A fixed random seed ensures reproducibility across runs.

---

## Part II – Model Training

### Model Selection

A **Logistic Regression** classifier is used as the baseline model due to:

- Efficiency on high-dimensional sparse data (TF-IDF)
- Strong performance on text classification tasks
- Interpretability of feature weights
- Stability for large-scale deployment

### Training Process (`train.py`)

The training pipeline performs the following steps:

1. Loads processed feature datasets from Azure ML datastore
2. Combines all feature sources into a single matrix
3. Trains a Logistic Regression model
4. Evaluates performance on validation and test sets
5. Logs metrics and artifacts to MLflow

**Metrics tracked:** Accuracy, Precision, Recall, F1-score

### MLflow Experiment Tracking

All training runs are tracked using MLflow integrated with Azure ML, enabling:

- Comparison of multiple runs
- Logging of hyperparameters (`C`, `max_iter`)
- Model artifact storage
- Reproducibility of experiments

---

## Part III – Azure ML Pipeline Architecture

### Pipeline Structure

The project follows a modular Azure ML pipeline design where each stage is isolated to ensure independent execution, easy debugging, and reusability across experiments.

### Training Job Submission

The training job is submitted to Azure ML using a CLI-driven job definition (`train_job.yml`). This job:

- Runs on Azure compute cluster
- Executes `train.py`
- Logs outputs to MLflow
- Stores trained model as `model.pkl`

### Hyperparameter Sweep

A sweep job (`sweep_job.yml`) optimizes model performance by tuning:

| Parameter | Description |
|---|---|
| `C` | Regularization strength |
| `max_iter` | Maximum iterations |

The sweep uses:
- Random sampling strategy
- Accuracy as the optimization metric
- Multiple concurrent trials for efficiency

---

## Part IV – Model Deployment

### Scoring Script (`score.py`)

The deployed model uses a custom scoring script that:

- Loads the trained model from Azure ML model registry
- Accepts JSON input via REST API
- Performs batch prediction
- Returns sentiment predictions as JSON output

### Inference Endpoint

The trained model is deployed as a **Managed Online Endpoint** on Azure ML with:

- REST API access
- Scalable compute instance
- Real-time prediction support
- Secure authentication via API keys

### Endpoint Testing (`invoke_endpoint.py`)

A Python-based client script verifies end-to-end deployment:
```python
# Loads sample input data
# Sends HTTP POST request to endpoint
# Receives and prints predictions
```

---

## Part V – CI/CD with Azure DevOps

A CI/CD pipeline (`azure-pipelines.yml`) automates the full training workflow. On every push to the repository:

1. Azure CLI installs ML extension
2. Workspace and resource group are configured
3. Training job is submitted automatically
4. Logs are streamed in real time

This ensures continuous training, reproducibility, and full automation of the ML workflow.

---

## Part VI – Results & Evaluation

### Model Performance

| Metric | Result |
|---|---|
| Accuracy | High (validated on test set) |
| Precision | Strong class separation |
| Recall | Balanced across sentiment classes |
| F1-score | Consistent across folds |

### Key Observations

- TF-IDF captures strong surface-level sentiment indicators
- SBERT embeddings improve semantic generalization
- Logistic Regression performs well despite its simplicity
- Hyperparameter tuning provides marginal but consistent gains
- CI/CD pipeline ensures reproducible training on every commit

---

## Part VII – Challenges & Solutions

### Data Inconsistencies in Label Propagation

The pipeline initially failed because the expected label column (`overall`) was missing after feature merging. A temporary fallback mechanism was introduced to allow pipeline execution while isolating the root cause.

**Final fix:** Ensuring correct label alignment at the feature merging stage so that sentiment labels are preserved consistently across all splits.

### Feature Integration Complexity

Combining TF-IDF, SBERT embeddings, and statistical features resulted in high-dimensional matrices requiring careful index alignment. Resolved by:

- Ensuring consistent indexing before concatenation
- Standardizing feature outputs as NumPy arrays / DataFrames
- Validating shape consistency before model training

### Azure ML Job Debugging & Execution Delays

Job submission failures occurred due to incorrect dataset asset references, missing environment dependencies, and CLI extension version mismatches. Resolved by:

- Standardizing dataset naming conventions
- Locking environment dependencies in `conda.yml`
- Updating the Azure ML CLI extension before job submission

### MLflow Tracking Configuration Issues

Initial MLflow runs failed due to incompatible tracking URIs when using standard `mlflow` instead of `azureml-mlflow`.

**Fix:** Switched to `azureml-mlflow` for Azure-native tracking support, ensuring seamless experiment logging and model registry integration.

---

## Part VIII – Versioning & Best Practices

### Code & Experiment Versioning

- **Git-based version control** for all source code, configuration files, and pipeline definitions
- Semantic commit messages (`feat`, `fix`, `chore`) for full traceability
- Separate commits maintained for: training logic, pipeline config, environment updates, and deployment scripts

### Data & Model Versioning

- Azure ML dataset versioning tracks different stages of processed data
- Each dataset (train/validation/test) registered as a versioned Azure ML asset
- Models stored in the **Azure ML Model Registry** for version tracking, rollback, and run comparison

### Experiment Tracking (MLflow)

MLflow tracks all of the following per run:

- Hyperparameters (`C`, `max_iter`)
- Performance metrics (accuracy, F1-score, etc.)
- Model artifacts (`model.pkl`)
- Run-level metadata

### CI/CD Best Practices

- Automated training triggered on every commit — no manual intervention required
- Azure CLI used for reproducible job execution
- Environment isolation for consistent runtime behavior

### General MLOps Best Practices

- Modular pipeline design (separation of training, preprocessing, deployment)
- Environment isolation using Conda YAML files
- Reproducible experiments using fixed random seeds
- Strict train/validation/test separation to avoid data leakage
- Logging and monitoring via MLflow integration

---

## Bonus: Known Limitation

> **What is one thing being done "not correctly" in this assignment?**

Feature engineering (TF-IDF / SBERT / transformations) is performed **before** splitting the dataset into training and testing sets, which leads to **data leakage**. The correct approach is to fit all transformers exclusively on the training set and apply them to the validation/test sets only at inference time.

---

## Component Summary

| Component | Purpose | Output |
|---|---|---|
| `train.py` | Model training | Logistic Regression model |
| `sweep_job.yml` | Hyperparameter tuning | Best `C` and `max_iter` |
| `score.py` | Inference logic | REST API predictions |
| `deployment.yml` | Model deployment | Managed endpoint |
| `azure-pipelines.yml` | CI/CD automation | Auto training pipeline |
