# ============================================================
# Feature Extraction Component
# Extracts time-series features using tsfresh
# ============================================================

import argparse
import os
import time
import json
import pandas as pd
import numpy as np
from tsfresh import extract_features
from tsfresh.feature_extraction import (
    MinimalFCParameters,
    EfficientFCParameters,
    ComprehensiveFCParameters
)
import mlflow

def parse_args():
    parser = argparse.ArgumentParser("extract_features")
    parser.add_argument("--train_data", type=str, help="Path to tsfresh-ready training data")
    parser.add_argument("--test_data", type=str, help="Path to tsfresh-ready test data")
    parser.add_argument("--rul_labels", type=str, help="Path to RUL labels")
    parser.add_argument("--feature_set", type=str, default="MinimalFCParameters")
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--train_features", type=str, help="Output path for train features")
    parser.add_argument("--test_features", type=str, help="Output path for test features")
    parser.add_argument("--extraction_metrics", type=str, help="Output path for metrics")
    return parser.parse_args()

def get_feature_set(feature_set_name):
    """Return tsfresh feature set based on name"""
    feature_sets = {
        "MinimalFCParameters": MinimalFCParameters(),
        "EfficientFCParameters": EfficientFCParameters(),
        "ComprehensiveFCParameters": ComprehensiveFCParameters()
    }
    if feature_set_name not in feature_sets:
        print(f"Unknown feature set {feature_set_name}, defaulting to MinimalFCParameters")
        return MinimalFCParameters()
    return feature_sets[feature_set_name]

def load_parquet_folder(path):
    """Load all parquet files from a folder into a single dataframe"""
    files = [f for f in os.listdir(path) if f.endswith(".parquet")]
    dfs = [pd.read_parquet(os.path.join(path, f)) for f in files]
    return pd.concat(dfs, ignore_index=True)

def extract_tsfresh_features(df, feature_set, n_jobs, dataset_name):
    """Extract tsfresh features from dataframe"""
    print(f"\n=== Extracting features for {dataset_name} ===")
    print(f"Input shape   : {df.shape}")
    print(f"Engines       : {df['engine_id'].nunique()}")
    print(f"Feature set   : {feature_set.__class__.__name__}")
    print(f"Parallel jobs : {n_jobs}")

    # Get sensor columns only
    sensor_cols = [c for c in df.columns
                   if c not in ["engine_id", "cycle"]]

    print(f"Sensor cols   : {len(sensor_cols)}")

    start_time = time.time()

    # Extract features per engine
    features_df = extract_features(
        df,
        column_id="engine_id",
        column_sort="cycle",
        default_fc_parameters=feature_set,
        n_jobs=n_jobs,
        disable_progressbar=False
    )

    elapsed = time.time() - start_time
    print(f"\nExtraction complete")
    print(f"Output shape  : {features_df.shape}")
    print(f"Runtime       : {elapsed:.2f} seconds")

    return features_df, elapsed

def main():
    args = parse_args()

    # Start MLflow run for tracking
    mlflow.start_run()

    print("=" * 50)
    print("FEATURE EXTRACTION COMPONENT")
    print("=" * 50)

    # ── Load Data ────────────────────────────────────────────
    print("\nLoading data...")
    train_df = load_parquet_folder(args.train_data)
    test_df = load_parquet_folder(args.test_data)
    labels_df = load_parquet_folder(args.rul_labels)

    print(f"Train loaded  : {train_df.shape}")
    print(f"Test loaded   : {test_df.shape}")
    print(f"Labels loaded : {labels_df.shape}")

    # ── Get Feature Set ──────────────────────────────────────
    feature_set = get_feature_set(args.feature_set)

    # ── Extract Features ─────────────────────────────────────
    train_features, train_time = extract_tsfresh_features(
        train_df, feature_set, args.n_jobs, "TRAINING"
    )
    test_features, test_time = extract_tsfresh_features(
        test_df, feature_set, args.n_jobs, "TEST"
    )

    # ── Attach RUL Labels to Train Features ──────────────────
    # labels_df has engine_id + cycle + target_RUL
    # train_features index is engine_id
    # We need to merge labels with features

    # Get one label per engine — use median RUL as representative label
    engine_labels = labels_df.groupby("engine_id")["target_RUL"].median().reset_index()
    engine_labels.columns = ["engine_id", "target_RUL"]
    engine_labels = engine_labels.set_index("engine_id")

    # Merge labels with features
    train_features = train_features.join(engine_labels, how="left")

    print(f"\n=== FINAL FEATURE SHAPES ===")
    print(f"Train features: {train_features.shape}")
    print(f"Test features : {test_features.shape}")
    print(f"RUL column    : {'target_RUL' in train_features.columns}")

    # ── Handle NaN/Inf values ────────────────────────────────
    # tsfresh can produce NaN/Inf for some features
    train_features = train_features.replace([np.inf, -np.inf], np.nan)
    test_features = test_features.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with column median
    train_features = train_features.fillna(train_features.median())
    test_features = test_features.fillna(test_features.median())

    print(f"\nNaN/Inf values handled")

    # ── Log Metrics to MLflow ────────────────────────────────
    total_time = train_time + test_time
    n_features = train_features.shape[1] - 1  # exclude target_RUL

    mlflow.log_metric("train_extraction_time_seconds", train_time)
    mlflow.log_metric("test_extraction_time_seconds", test_time)
    mlflow.log_metric("total_extraction_time_seconds", total_time)
    mlflow.log_metric("n_extracted_features", n_features)
    mlflow.log_param("feature_set", args.feature_set)
    mlflow.log_param("n_jobs", args.n_jobs)

    print(f"\n=== EXTRACTION METRICS ===")
    print(f"Total features extracted : {n_features}")
    print(f"Total runtime            : {total_time:.2f} seconds")

    # ── Save Outputs ─────────────────────────────────────────
    os.makedirs(args.train_features, exist_ok=True)
    os.makedirs(args.test_features, exist_ok=True)
    os.makedirs(args.extraction_metrics, exist_ok=True)

    train_features.to_parquet(
        os.path.join(args.train_features, "train_features.parquet")
    )
    test_features.to_parquet(
        os.path.join(args.test_features, "test_features.parquet")
    )

    # Save metrics as JSON
    metrics = {
        "feature_set": args.feature_set,
        "n_jobs": args.n_jobs,
        "n_extracted_features": n_features,
        "train_extraction_time_seconds": round(train_time, 2),
        "test_extraction_time_seconds": round(test_time, 2),
        "total_extraction_time_seconds": round(total_time, 2),
        "train_shape": list(train_features.shape),
        "test_shape": list(test_features.shape)
    }

    with open(os.path.join(args.extraction_metrics, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nOutputs saved successfully")
    print(f"Train features : {args.train_features}")
    print(f"Test features  : {args.test_features}")
    print(f"Metrics        : {args.extraction_metrics}")

    mlflow.end_run()
    print("\nFeature extraction component complete!")

if __name__ == "__main__":
    main()
