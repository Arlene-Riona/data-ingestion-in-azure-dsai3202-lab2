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
- Sentiment label derived from the `overall` rating column

### Feature Engineering Pipeline

Raw text is transformed into ML-ready features using:

| Feature Type | Description |
|---|---|
| TF-IDF Vectorization | Captures word importance across documents |
| Sentence-BERT Embeddings | Captures semantic meaning |
| Statistical Features | Review length, word count, avg word length, punctuation frequency |
| Sentiment Lexicon Signals | VADER-based features |

These features are combined into a unified feature matrix used for model training.

### Dataset Split Strategy

This assignment introduced a four-way dataset partition, replacing the original three-split approach from Lab 4. The updated split strategy is:

| Split | Proportion | Purpose |
|---|---|---|
| **Train** | 60% | What the model actually learns from |
| **Validation** | 15% | Hyperparameter tuning and experiment comparison |
| **Test (Holdout)** | 15% | Final, unbiased offline evaluation |
| **Deployment** | 10% | Simulates incoming production data post-deployment |

The deployment split is sourced from the most recent time period in the dataset (based on `review_year`), intentionally reflecting potential data drift relative to the training distribution. This design mirrors real-world production conditions where language evolves, products change, and review behavior shifts over time.

Each split was passed through the full feature engineering pipeline so that all four datasets contain the same engineered features: SBERT embeddings, TF-IDF vectors, sentiment scores, length statistics, and the `overall` label column.

<img width="1857" height="378" alt="image" src="https://github.com/user-attachments/assets/706a1751-0f3a-4c78-8fd0-d9ade3b6114a" />

---

## Part II – Model Training

### Model Selection

A **Logistic Regression** classifier is used as the primary model due to:

- Efficiency on high-dimensional sparse data (TF-IDF)
- Strong performance on text classification tasks
- Interpretability of feature weights
- Stability for large-scale deployment

### Training Process (`train.py`)

The training pipeline performs the following steps:

1. Loads processed feature datasets from Azure ML datastore
2. Combines all feature sources into a single matrix
3. Trains a Logistic Regression model
4. Evaluates performance on training, validation, and test sets
5. Logs metrics and artifacts to MLflow

### MLflow Metric Logging

All training runs are tracked using MLflow integrated with Azure ML. The following metrics are logged per run:

| Metric | Train | Validation | Test |
|---|---|---|---|
| Accuracy | 0.8484 | 0.8404 | 0.8422 |
| AUC | 0.8499 | 0.8154 | 0.8182 |
| F1 Score | 0.9101 | 0.9060 | 0.9073 |
| Precision | 0.8681 | 0.8618 | 0.8633 |
| Recall | 0.9564 | 0.9550 | 0.9561 |

Total pipeline runtime was also logged via MLflow: **70.18 seconds**.

In addition to metrics, the hyperparameters (`C`, `max_iter`) used per run are logged, enabling full reproducibility and comparison across experiments.

---

## Part III – Azure ML Pipeline Architecture

### Pipeline Structure

The project follows a modular Azure ML pipeline design where each stage is isolated to ensure independent execution, easy debugging, and reusability across experiments.

### Feature Engineering Pipeline (Lab 4 — Re-run)

Lab 4 was revisited to introduce the deployment split and to ensure all four dataset partitions contain the full set of engineered features and the `overall` label column. The primary aim of re-running Lab 4 was to produce the new deployment split; the `overall` label fix was a side effect of correcting the feature pipeline output. All four merged datasets were registered as versioned Azure ML Data Assets:

- `amazon_review_merged_features_train`
- `amazon_review_merged_features_val`
- `amazon_review_merged_features_test`
- `amazon_review_merged_features_deploy`

### Training Job Submission

The training job is submitted to Azure ML using a CLI-driven job definition (`train_job.yml`). This job:

- Runs on an Azure ML CPU compute cluster
- Executes `train.py` with explicit `--resource-group` and `--workspace-name` flags
- Logs outputs to MLflow
- Stores the trained model as `model.pkl`

### Hyperparameter Sweep

A sweep job (`sweep_job.yml`) optimizes model performance by tuning:

| Parameter | Description |
|---|---|
| `C` | Regularization strength |
| `max_iter` | Maximum iterations |

The sweep uses a random sampling strategy, optimizes for validation accuracy, and runs multiple concurrent trials. The best configuration identified by the sweep was applied as the default in the final training run.

---

## Part IV – Model Deployment

### Scoring Script (`score.py`)

The deployed model uses a custom scoring script that:

- Loads the trained model from Azure ML model registry
- Accepts JSON input via REST API
- Performs batch prediction using the same feature construction logic as training
- Returns sentiment predictions as JSON output

### Inference Endpoint

The trained model is deployed as a **Managed Online Endpoint** on Azure ML with:

- REST API access
- Scalable compute instance (`Standard_F2s_v2`)
- Real-time prediction support
- Secure authentication via API keys

### Endpoint Testing (`invoke_endpoint.py`)

A Python-based client script verifies end-to-end deployment by loading the deployment dataset (the 10% production-simulation split), constructing the feature matrix, sending HTTP POST requests to the endpoint, and computing evaluation metrics against the true labels.

---

## Part V – CI/CD with Azure DevOps

A CI/CD pipeline (`azure-pipelines.yml`) automates the full training workflow on every push to the `assignment2_model_training` branch. The pipeline:

1. Installs the Azure ML CLI extension
2. Authenticates with Azure via the `SC-UDST-CCIT-DSAI3202` service connection
3. Submits the training job with explicit `--resource-group` and `--workspace-name` flags passed directly to the `az ml` commands
4. Streams job logs in real time

The DevOps pipeline configuration encountered a service connection issue during setup, which was resolved by ensuring the service connection name in the YAML matched exactly the name registered in Azure DevOps Project Settings, and by passing resource group and workspace parameters explicitly rather than relying on `az configure --defaults`, which does not propagate reliably to the `ml` CLI extension.

---

## Part VI – Results & Evaluation

### Model Performance

| Metric | Train | Validation | Test |
|---|---|---|---|
| Accuracy | 0.8484 | 0.8404 | 0.8422 |
| AUC | 0.8499 | 0.8154 | 0.8182 |
| F1 Score | 0.9101 | 0.9060 | 0.9073 |
| Precision | 0.8681 | 0.8618 | 0.8633 |
| Recall | 0.9564 | 0.9550 | 0.9561 |

**Pipeline runtime:** 70.18 seconds

### Key Observations

- TF-IDF captures strong surface-level sentiment indicators
- SBERT embeddings improve semantic generalization
- Logistic Regression performs well despite its simplicity, with consistent metrics across all splits
- The close alignment between validation and test scores indicates good generalization without overfitting
- Hyperparameter tuning via sweep jobs provided consistent marginal improvements
- The deployment split, sourced from the most recent reviews, enables observation of potential data drift in production

---

## Part VII – Challenges & Solutions

### Introducing the Deployment Split (Lab 4 Re-run)

The original Lab 4 pipeline only produced three dataset splits. To meet the updated requirements, the splitting logic was revised to produce a fourth deployment split (10%), drawn from the most recent reviews by `review_year`. The full feature engineering pipeline was then re-run so that all four splits contain identical feature columns and the `overall` label.

### `overall` Label Preservation

Some pipeline configurations had accidentally dropped the `overall` rating column during feature merging, which caused the training script to fail at label creation. This was resolved by ensuring the merge logic explicitly preserved the label column across all dataset splits.

### Feature Integration Complexity

Combining TF-IDF, SBERT embeddings, and statistical features resulted in high-dimensional matrices requiring careful index alignment. This was resolved by ensuring consistent indexing before concatenation, standardizing feature outputs as NumPy arrays, and validating shape consistency before model training.

### Azure ML CLI and `az configure --defaults`

The `az configure --defaults` command does not reliably propagate to the `az ml` CLI extension, causing job submissions to fail with a missing `--resource-group` argument error. The fix was to pass `--resource-group` and `--workspace-name` explicitly on every `az ml` command rather than relying on global defaults.

### MLflow Tracking Configuration

Initial MLflow runs failed due to incompatible tracking URIs when using the standard `mlflow` package instead of `azureml-mlflow`. Switching to `azureml-mlflow` resolved the issue, enabling seamless experiment logging and model registry integration with Azure ML.

---

## Part VIII – Versioning & Best Practices

### Branch Strategy

Rather than modifying the original Lab 4 branch directly, a dedicated branch was created from it for the required changes. This preserved the original pipeline state and kept modifications isolated and traceable — a standard practice for production ML workflows where changes to upstream pipeline stages need to be auditable.

### Commit Message Conventions

Commits followed semantic prefixes throughout the project (`feat`, `fix`, `chore`) to maintain full traceability across training logic changes, pipeline configuration updates, environment definition changes, and deployment scripts. This makes the history readable and clearly separates concern areas at a glance.

### Data & Model Versioning

- Azure ML dataset versioning tracks different stages of processed data across pipeline re-runs
- Each dataset (train/validation/test/deploy) is registered as a versioned Azure ML asset using `azureml:<asset_name>@latest`
- Models are stored in the **Azure ML Model Registry** for version tracking, rollback capability, and run comparison

### Experiment Tracking (MLflow)

MLflow tracks the following per run:

- Hyperparameters (`C`, `max_iter`)
- Performance metrics (accuracy, AUC, F1, precision, recall) across all splits
- Model artifact (`model.pkl`)
- Total pipeline runtime

### General MLOps Best Practices

- Modular pipeline design with clear separation between feature engineering, training, and deployment
- Separate conda environments for training (`conda.yml`) and inference (`inference_conda.yml`), keeping each minimal to reduce Docker build time
- Reproducible experiments using fixed random seeds
- Strict train/validation/test/deployment separation to minimize data leakage
- CI/CD automation ensures every push to the trigger branch results in a reproducible training run with no manual intervention

---

## Bonus: Known Limitation

> **What is one thing being done "not correctly" in this assignment?**

In the assignment, during the hyperparameter sweep, the test set is evaluated and its metrics are logged during every single training run, including every child run in the hyperparameter sweep. Here the test set is repeated exposure to test metrics which violates the holdout principle as seeing the test set performance across every trial can influence decisions about which configurations to select, which features to include, or which model to use, making the final reported test accuracy optimistically biased and no longer a reliable estimate of true generalization performance. The correct approach would be to remove test set evaluation from the training script entirely during the experimentation phase, evaluate it only once after the best model has been selected based solely on validation metrics.
