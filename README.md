# Lab 5 – Scalable Feature Extraction and Selection for Predictive Maintenance

## Overview

This lab builds a scalable end-to-end machine learning pipeline for **Remaining Useful Life (RUL) prediction** of aircraft turbofan engines using the NASA C-MAPSS FD001 dataset. The goal is to predict how many cycles remain before an engine fails based on sensor readings, which is a core problem in predictive maintenance.

The pipeline follows a **medallion architecture** (Bronze → Silver → Gold) on Azure Blob Storage, uses **Azure Databricks** for ETL processing, **tsfresh** for automated time-series feature extraction using rolling windows, a **DEAP genetic algorithm** for optimal feature subset selection, and an **Azure ML Pipeline** with 5 modular command components for scalable and reproducible model training and evaluation.

---

## Part I – Medallion Architecture and Databricks ETL

### 1. Storage Setup and Medallion Architecture

Data was organised across three layers in Azure Blob Storage, following the medallion architecture pattern where each layer represents a progressively cleaner and more enriched version of the data:

- **Bronze layer** (`raw/FD001/`) — raw `.txt` files exactly as received from NASA, never modified. This ensures we can always trace back to the original source.
- **Silver layer** (`processed/FD001/`) — cleaned, correctly typed, and validated data written as Parquet. Column names are assigned, types are cast, and constant sensors are removed.
- **Gold layer** (`curated/FD001/`) — fully normalised, RUL-labelled, tsfresh-ready data for ML consumption.

This separation means that if any downstream step needs to be re-run or changed, only the relevant layer needs to be reprocessed without touching the earlier ones.

---

### 2. Bronze → Silver (`01_bronze_ingestion`)

The raw C-MAPSS files have no column headers and contain trailing empty columns. The first Databricks notebook reads the raw training and test files, assigns all 26 correct column names, casts `engine_id` and `cycle` to integers and all sensor and operational setting columns to floats, and drops the two trailing empty columns.

**Constant sensor removal:** Seven sensors (`sensor_1`, `sensor_5`, `sensor_6`, `sensor_10`, `sensor_16`, `sensor_18`, `sensor_19`) were found to have zero variance across all engines and cycles. These carry no information for RUL prediction and are removed at this stage to reduce noise and downstream computation.

Output written to the Silver layer as Parquet:
- Train: 20,631 rows, 19 columns
- Test: 13,096 rows, 19 columns

---

### 3. Silver → Gold (`02_silver_to_gold`)

The second notebook enriches the silver data with two key transformations:

**RUL Computation:** For each engine, RUL at each cycle is computed as `max_cycle − current_cycle`, giving the ground truth label for how many cycles remain before failure.

**RUL Clipping at 125:** Raw RUL values can be very large for engines that ran a long time. However, turbofan sensors only show measurable degradation in approximately the last 125 cycles. Clipping at 125 implements the standard **piecewise linear degradation assumption** used across C-MAPSS benchmarks — it focuses the model on the degradation phase and ignores the early healthy phase where all sensors look identical regardless of engine health.

**Sensor Normalisation:** All 14 remaining sensor columns and 3 operational setting columns are scaled to [0, 1] using MinMaxScaler. The scaler is fitted on the training set only and then applied to both train and test to prevent data leakage. Normalised columns are written with a `_scaled` suffix.

---

### 4. Gold → tsfresh-Ready (`03_gold_to_features`)

The third notebook prepares the data for tsfresh ingestion by separating feature columns from RUL labels and writing three outputs to the Gold layer:

- `curated/FD001/train_tsfresh_ready` — sensor features for all 20,631 training rows
- `curated/FD001/test_tsfresh_ready` — sensor features for all 13,096 test rows
- `curated/FD001/train_rul_labels` — per-cycle RUL labels (`engine_id`, `cycle`, `target_RUL`)

Label statistics confirmed correct computation: mean RUL = 86.83, min = 0, max = 125, stddev = 41.67.

---

## Part II – Azure ML Pipeline

### 5. Repository Structure and Component Setup

The repository was structured to cleanly separate Databricks notebooks, Azure ML components, pipeline definitions, and configuration. Each component lives in its own folder containing its Python script, `component.yml`, and `conda.yml` to keep dependencies isolated and independently versioned. The `.env` file containing Azure credentials is gitignored and a `.env.example` template is committed instead.
```
repo/
├── databricks/
│   ├── 01_bronze_to_silver.ipynb
│   ├── 02_silver_to_gold.ipynb
│   └── 03_gold_to_features.ipynb
├── components/
│   ├── extract_features/
│   │   ├── component.yml
│   │   ├── conda.yml
│   │   └── extract_features.py
│   ├── filter_selection/
│   │   ├── component.yml
│   │   ├── conda.yml
│   │   └── filter_selection.py
│   ├── genetic_algorithm/
│   │   ├── component.yml
│   │   ├── conda.yml
│   │   └── genetic_algorithm.py
│   ├── split_dataset/
│   │   ├── component.yml
│   │   ├── conda.yml
│   │   └── split_dataset.py
│   └── train_evaluate/
│       ├── component.yml
│       ├── conda.yml
│       └── train_evaluate.py
├── config/config.yaml
├── data/cmapss_fd001.yml
├── pipelines/feature_pipeline.yml
├── .env.example
└── README.md
```

The screenshot below shows the component directories being created in PowerShell using `New-Item`:

<img width="940" alt="Creating the GitHub repository structure" src="screenshot_01_repo_structure.png" />

---

### 6. Registering Azure ML Components

Each of the 5 pipeline components was registered in Azure ML using the Azure CLI. Registering components separately from the pipeline means they can be versioned, reused across pipelines, and updated independently. The `code: .` field in each `component.yml` tells Azure ML to upload the local script directory as the component's source code snapshot — without this, Azure ML cannot find the Python script at runtime and throws a `No such file or directory` error.

All `conda.yml` files use `azureml-mlflow` instead of the generic `mlflow` package. The standard `mlflow` package does not understand `azureml://` tracking URIs used natively by Azure ML and raises an `UnsupportedModelRegistryStoreURIException`. Replacing it with `azureml-mlflow` resolves this with no code changes required.

#### extract_features

Registered with inputs for `train_data`, `test_data`, `rul_labels`, `feature_set`, and `n_jobs`, and outputs for `train_features`, `test_features`, and `extraction_metrics`.

<img width="940" alt="Registering extract_features component" src="screenshot_02_register_extract_features.png" />

---

#### filter_selection

Registered with configurable thresholds for variance, correlation, and mutual information filtering. The registered command confirms all expected inputs and outputs including `variance_threshold`, `correlation_threshold`, and `mutual_info_percentile`.

<img width="940" alt="Registering filter_selection component" src="screenshot_03_register_filter_selection.png" />

---

#### genetic_algorithm

Registered with all DEAP hyperparameters exposed as inputs — `population_size`, `n_generations`, `crossover_prob`, `mutation_prob`, `tournament_size`, and `min_features` — confirming the component is fully configurable without code changes.

<img width="940" alt="Registering genetic_algorithm component" src="screenshot_04_register_genetic_algorithm.png" />

---

#### split_dataset

Registered with `test_size` and `random_seed` as configurable inputs to ensure reproducible splits.

<img width="940" alt="Registering split_dataset component" src="screenshot_05_register_split_dataset.png" />

---

#### train_evaluate

Registered with `model_type`, `n_estimators`, and `max_depth` as configurable inputs, allowing different model types to be tested without modifying any code.

<img width="940" alt="Registering train_evaluate component" src="screenshot_06_register_train_evaluate.png" />

---

### 7. Pipeline Components — What Each Step Does

#### Extract Features (`extract_features`)

Uses tsfresh to extract statistical time-series features from the sensor data using a **rolling window approach**.

**Why rolling windows?** A naive approach of extracting one feature vector per engine produces only 100 rows and attaches a single RUL label per engine. Due to RUL clipping at 125, nearly all engines get the same label for most of their life, leaving almost no variation for the model to learn from. The rolling window approach extracts one feature vector per `(engine, cycle)` pair using the most recent 30 cycles as the input window, stepped every 5 cycles. This produces ~4,000 rows with properly varying RUL labels from 0 to 125 at each time step.

`MinimalFCParameters` was chosen over `EfficientFCParameters` because the efficient set (~700 features) exceeded 1 hour of runtime on the Standard_DS3_v2 instance, while the minimal set produces comparable accuracy in minutes.

| Parameter | Value |
|---|---|
| Feature set | `MinimalFCParameters` |
| Window size | 30 cycles |
| Step size | 5 cycles |
| Output rows (train) | ~4,063 |

---

#### Filter-Based Feature Selection (`filter_selection`)

Three sequential filters reduce the feature space before the expensive genetic algorithm runs. Running filters first shrinks the search space significantly, making the GA faster and reducing the risk of selecting noise features.

| Step | Method | Threshold | Purpose |
|---|---|---|---|
| 1 | Variance threshold | < 0.01 | Remove near-constant features |
| 2 | Correlation filter | > 0.95 | Remove redundant duplicate features |
| 3 | Mutual information | Bottom 50% | Keep features most informative about RUL |

---

#### Genetic Algorithm Feature Selection (`genetic_algorithm`)

Uses **DEAP** (Distributed Evolutionary Algorithms in Python) to search for the optimal feature subset. Each individual in the population is a binary chromosome where `1` = include feature and `0` = exclude. Fitness is evaluated using 3-fold cross-validated RMSE with a lightweight RandomForest (20 estimators) to balance evaluation speed with accuracy.

The GA is preferred over purely filter-based methods because it can discover **feature combinations** that work well together, not just features that are individually strong.

| Parameter | Value |
|---|---|
| Population size | 50 |
| Generations | 20 |
| Crossover probability | 0.7 |
| Mutation probability | 0.2 |
| Selection | Tournament |
| Fitness | 3-fold CV RMSE |

---

#### Split Dataset (`split_dataset`)

Splits the GA-selected training features into 80% training and 20% validation with a fixed random seed. The split occurs after all feature selection steps to ensure no validation data influenced which features were selected or how they were extracted.

---

#### Train and Evaluate (`train_evaluate`)

Trains a **RandomForest Regressor** on the training split and evaluates on the validation split. The model is saved as `model.pkl` via joblib, predictions are saved as Parquet and CSV, and all metrics are written to `evaluation_metrics.json`.

| Parameter | Value |
|---|---|
| Model | RandomForest |
| n_estimators | 100 |
| max_depth | 10 |

---

### 8. Running the Pipeline

All 5 components were wired together in `pipelines/feature_pipeline.yml` and submitted to the Azure ML compute instance `azurecompute60304739` (Standard_DS3_v2). The pipeline runs entirely on Azure ML compute, not locally, and each step's inputs, outputs, and metrics are automatically tracked and versioned in the workspace.

The `name:` field was removed from the pipeline YAML so Azure ML auto-generates a unique run name each submission, avoiding the `A job with this name already exists` error when resubmitting after fixes.
```powershell
Get-Content .env | ForEach-Object {
  if ($_ -match '^([^#][^=]*)=(.*)$') {
    [System.Environment]::SetEnvironmentVariable($matches[1], $matches[2])
  }
}

az ml job create --file pipelines/feature_pipeline.yml `
  --workspace-name $env:AZURE_WORKSPACE_NAME `
  --resource-group $env:AZURE_RESOURCE_GROUP `
  --stream
```

All 5 components completed successfully as shown below:

<img width="940" alt="Completed pipeline in Azure ML Studio — all 5 components green" src="screenshot_07_pipeline_success.png" />

---

## Part III – Results

### 9. Validation Metrics

| Metric | Train | Validation |
|---|---|---|
| RMSE | 9.66 | **14.38** |
| MAE | 6.51 | **9.85** |
| R² | 0.9465 | **0.8794** |

A validation R² of 0.88 means the model explains 88% of the variance in RUL — a strong result for a Random Forest with only 6 features and no hyperparameter tuning beyond defaults.

---

### 10. Top Features Selected by Genetic Algorithm

| Rank | Feature | Importance |
|---|---|---|
| 1 | `sensor_2_scaled__sum_values` | 65.3% |
| 2 | `cycle` | 16.0% |
| 3 | `sensor_4_scaled__maximum` | 8.2% |
| 4 | `sensor_15_scaled__maximum` | 3.8% |
| 5 | `sensor_7_scaled__root_mean_square` | 3.7% |
| 6 | `sensor_3_scaled__absolute_maximum` | 3.1% |

`sensor_2` (total temperature at LPC outlet) and `sensor_4` (total temperature at HPC outlet) are well-established degradation indicators in the C-MAPSS FD001 literature. Their dominance in feature importance confirms the pipeline selected physically meaningful signals rather than noise.

---

### 11. Test Set Predictions

| Statistic | Value |
|---|---|
| Mean predicted RUL | 104.43 cycles |
| Min predicted RUL | 9.55 cycles |
| Max predicted RUL | 125.00 cycles |

The spread of predictions from ~10 to 125 confirms the model produces meaningful, varied RUL estimates. This was a key failure mode during early development — before the rolling window fix, all predictions were constant at 125 due to incorrect label aggregation.

---

## Summary

| Component | Purpose | Key Output |
|---|---|---|
| `extract_features` | Rolling window tsfresh extraction | ~4,063 rows × ~10 features |
| `filter_selection` | Variance, correlation, MI filters | Reduced feature set |
| `genetic_algorithm` | DEAP binary GA optimisation | 6 optimal features |
| `split_dataset` | 80/20 train/validation split | Train and val Parquet |
| `train_evaluate` | RandomForest training and evaluation | RMSE 14.38, R² 0.88 |
