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

## Part II – Time-Series Exploration and Validation

### 1. Engine Lifetime Distribution
The training set contains 100 engines with lifetimes ranging from 128 to 362 cycles. 
The majority of engines lived well beyond the RUL clip threshold of 125, justifying 
the piecewise linear degradation assumption used in preprocessing.

<img width="987" height="487" alt="image" src="https://github.com/user-attachments/assets/98dd8c16-366d-4609-9325-d5a530c7ab38" />


---

### 2. RUL Distribution Before and After Clipping
Before clipping, RUL is heavily skewed with many values above 125. After clipping 
at 125, the distribution becomes uniform giving the model equal training examples 
at every degradation stage.

<img width="1186" height="493" alt="image" src="https://github.com/user-attachments/assets/71de6efa-6b6d-4e92-aae2-66f5f1b8aa01" />


---

### 3. Sensor Behaviour Across Engine Cycles
Several sensors show a clear monotonic trend as the engine approaches failure. 
sensor_2 and sensor_4 trend upward while sensor_7 trends downward, confirming 
they carry strong degradation signal across all engines.

<img width="1406" height="2552" alt="image" src="https://github.com/user-attachments/assets/91613c17-1b34-4602-9cbf-4903adbef486" />


---

### 4. Sensor Correlation with RUL
sensor_2, sensor_4, and sensor_11 show the strongest correlation with RUL. 
This directly motivates the mutual information filter in the pipeline which 
keeps only the top 50% most informative features.

<img width="986" height="587" alt="image" src="https://github.com/user-attachments/assets/c7cba25e-65e5-41fe-93fe-e15502ec8a11" />


---

### 5. Sensor Correlation Heatmap
Several sensor pairs exceed the 0.95 correlation threshold, confirming that 
redundant features exist in the dataset and justifying the correlation filter 
step in the pipeline.

<img width="1129" height="994" alt="image" src="https://github.com/user-attachments/assets/2766cb96-6c1b-4965-80db-bad9bcc72f6a" />


---

### 6. Sensor Readings by Life Phase
Sensors such as sensor_2 and sensor_4 show a clear difference between early 
life and end of life phases, confirming they carry strong degradation signal. 
This reinforces the feature selection decisions made by the genetic algorithm.

<img width="1387" height="983" alt="image" src="https://github.com/user-attachments/assets/aef26949-784e-48a8-a999-3058d6d9410f" />


---

## Part III – Azure ML Pipeline

### 5. Repository Structure and Component Setup

The repository was structured to cleanly separate Databricks notebooks, Azure ML components, pipeline definitions, and configuration. Each component lives in its own folder containing its Python script, `component.yml`, and `conda.yml` to keep dependencies isolated and independently versioned. The `.env` file containing Azure credentials is gitignored and a `.env.example` template is committed instead.
```
repo/
├── databricks/
│   ├── 01_bronze_ingestion.ipynb
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

<img width="940" height="278" alt="image" src="https://github.com/user-attachments/assets/8b91010f-2276-46d0-852b-c07bebad3d2c" />


---

### 6. Registering Azure ML Components

Each of the 5 pipeline components was registered in Azure ML using the Azure CLI. Registering components separately from the pipeline means they can be versioned, reused across pipelines, and updated independently. The `code: .` field in each `component.yml` tells Azure ML to upload the local script directory as the component's source code snapshot — without this, Azure ML cannot find the Python script at runtime and throws a `No such file or directory` error.

All `conda.yml` files use `azureml-mlflow` instead of the generic `mlflow` package. The standard `mlflow` package does not understand `azureml://` tracking URIs used natively by Azure ML and raises an `UnsupportedModelRegistryStoreURIException`. Replacing it with `azureml-mlflow` resolves this with no code changes required.

#### extract_features

Registered with inputs for `train_data`, `test_data`, `rul_labels`, `feature_set`, and `n_jobs`, and outputs for `train_features`, `test_features`, and `extraction_metrics`.

<img width="940" height="25" alt="image" src="https://github.com/user-attachments/assets/e0184a92-e107-46c4-8d11-0365edd6db81" />


---

#### filter_selection

Registered with configurable thresholds for variance, correlation, and mutual information filtering. The registered command confirms all expected inputs and outputs including `variance_threshold`, `correlation_threshold`, and `mutual_info_percentile`.

<img width="940" height="83" alt="image" src="https://github.com/user-attachments/assets/2259f094-d582-4181-87a6-6a86efd226a1" />


---

#### genetic_algorithm

Registered with all DEAP hyperparameters exposed as inputs — `population_size`, `n_generations`, `crossover_prob`, `mutation_prob`, `tournament_size`, and `min_features` — confirming the component is fully configurable without code changes.

<img width="940" height="92" alt="image" src="https://github.com/user-attachments/assets/e21d3754-4dda-4243-8bd4-5af53f727ca0" />


---

#### split_dataset

Registered with `test_size` and `random_seed` as configurable inputs to ensure reproducible splits.

<img width="940" height="80" alt="image" src="https://github.com/user-attachments/assets/af305687-b92e-4b2d-a9ec-b2aab165c933" />


---

#### train_evaluate

Registered with `model_type`, `n_estimators`, and `max_depth` as configurable inputs, allowing different model types to be tested without modifying any code.

<img width="940" height="86" alt="image" src="https://github.com/user-attachments/assets/c8002fcf-92b7-4b7d-b207-e097ef9269fe" />


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

<img width="1598" height="277" alt="image" src="https://github.com/user-attachments/assets/2370360a-8382-45d5-85b4-4d046bb5897a" />


All 5 components completed successfully as shown below:

<img width="700" height="700" alt="image" src="https://github.com/user-attachments/assets/22cfc750-1448-4aab-882a-9f1d18675fd1" />


---

## Part IV – Results

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
