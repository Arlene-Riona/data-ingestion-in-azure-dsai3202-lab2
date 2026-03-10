# ============================================================
# Feature Extraction Component
# Extracts time-series features using tsfresh with rolling windows
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
    parser.add_argument("--train_data", type=str)
    parser.add_argument("--test_data", type=str)
    parser.add_argument("--rul_labels", type=str)
    parser.add_argument("--feature_set", type=str, default="MinimalFCParameters")
    parser.add_argument("--n_jobs", type=int, default=4)
    parser.add_argument("--train_features", type=str)
    parser.add_argument("--test_features", type=str)
    parser.add_argument("--extraction_metrics", type=str)
    return parser.parse_args()

def get_feature_set(feature_set_name):
    """Return tsfresh feature set based on name"""
    feature_sets = {
        "MinimalFCParameters": MinimalFCParameters(),
        "EfficientFCParameters": EfficientFCParameters(),
        "ComprehensiveFCParameters": ComprehensiveFCParameters()
    }
    if feature_set_name not in feature_sets:
        print(f"Unknown feature set {feature_set_name}, defaulting to Minimal")
        return MinimalFCParameters()
    return feature_sets[feature_set_name]

def load_parquet_folder(path):
    """Load all parquet files from a folder"""
    files = [f for f in os.listdir(path) if f.endswith(".parquet")]
    dfs = [pd.read_parquet(os.path.join(path, f)) for f in files]
    return pd.concat(dfs, ignore_index=True)

def extract_windowed_features(df, labels_df, feature_set, dataset_name,
                               window_size=30, step=5):
    """
    Extract tsfresh features using rolling windows.
    
    For each engine at every `step` cycles:
    - Take last `window_size` cycles as input window
    - Extract tsfresh features from that window
    - Attach RUL label at that cycle
    
    This gives one row per (engine, cycle) with proper varying RUL labels.
    """
    print(f"\n=== Windowed Feature Extraction: {dataset_name} ===")
    print(f"Window size  : {window_size}")
    print(f"Step size    : {step} (extract every {step} cycles)")
    print(f"Engines      : {df['engine_id'].nunique()}")
    print(f"Total rows   : {df.shape[0]}")

    all_features = []
    engine_ids = sorted(df["engine_id"].unique())
    start_time = time.time()

    for i, engine_id in enumerate(engine_ids):
        if i % 10 == 0:
            print(f"Processing engine {i+1}/{len(engine_ids)}...")

        # Get all cycles for this engine
        engine_df = df[df["engine_id"] == engine_id].sort_values("cycle")
        cycles = engine_df["cycle"].values

        # Extract at every `step` cycles
        cycles_to_extract = cycles[::step]

        for cycle in cycles_to_extract:
            # Rolling window: last window_size cycles up to current cycle
            window = engine_df[
                engine_df["cycle"] <= cycle
            ].tail(window_size).copy()

            # Skip if window is too small
            if len(window) < 5:
                continue

            # tsfresh needs id column
            window = window.copy()
            window["ts_id"] = engine_id

            # Extract features for this window
            features = extract_features(
                window,
                column_id="ts_id",
                column_sort="cycle",
                default_fc_parameters=feature_set,
                n_jobs=1,
                disable_progressbar=True,
                show_warnings=False
            )

            # Add identifiers
            features["engine_id"] = engine_id
            features["cycle"] = cycle
            all_features.append(features)

    elapsed = time.time() - start_time

    if not all_features:
        raise ValueError("No features extracted! Check input data.")

    # Combine all windows
    features_df = pd.concat(all_features, ignore_index=True)

    print(f"\nExtraction complete")
    print(f"Output shape : {features_df.shape}")
    print(f"Runtime      : {elapsed:.2f}s")

    # ── Attach RUL Labels ────────────────────────────────────
    if labels_df is not None:
        print("\nAttaching RUL labels...")

        # labels_df has engine_id, cycle, target_RUL
        features_df = features_df.merge(
            labels_df[["engine_id", "cycle", "target_RUL"]],
            on=["engine_id", "cycle"],
            how="left"
        )

        # Check label attachment
        null_labels = features_df["target_RUL"].isnull().sum()
        if null_labels > 0:
            print(f"Warning: {null_labels} rows missing RUL labels")
            # Fill with nearest available label
            features_df["target_RUL"] = features_df.groupby("engine_id")[
                "target_RUL"
            ].transform(lambda x: x.fillna(method="ffill").fillna(method="bfill"))

        print(f"Label stats:")
        print(f"  Mean RUL : {features_df['target_RUL'].mean():.2f}")
        print(f"  Min RUL  : {features_df['target_RUL'].min():.2f}")
        print(f"  Max RUL  : {features_df['target_RUL'].max():.2f}")
        print(f"  Std RUL  : {features_df['target_RUL'].std():.2f}")

    return features_df, elapsed

def main():
    args = parse_args()

    mlflow.start_run()

    print("=" * 50)
    print("FEATURE EXTRACTION COMPONENT")
    print("=" * 50)
    print(f"Feature set  : {args.feature_set}")
    print(f"N jobs       : {args.n_jobs}")

    # ── Load Data ────────────────────────────────────────────
    print("\nLoading data...")
    train_df = load_parquet_folder(args.train_data)
    test_df = load_parquet_folder(args.test_data)
    labels_df = load_parquet_folder(args.rul_labels)

    print(f"Train loaded  : {train_df.shape}")
    print(f"Test loaded   : {test_df.shape}")
    print(f"Labels loaded : {labels_df.shape}")
    print(f"Label columns : {labels_df.columns.tolist()}")

    # ── Get Feature Set ──────────────────────────────────────
    feature_set = get_feature_set(args.feature_set)

    # ── Extract Features with Rolling Windows ────────────────
    train_features, train_time = extract_windowed_features(
        df=train_df,
        labels_df=labels_df,
        feature_set=feature_set,
        dataset_name="TRAINING",
        window_size=30,
        step=5
    )

    # Test data has no labels
    test_features, test_time = extract_windowed_features(
        df=test_df,
        labels_df=None,
        feature_set=feature_set,
        dataset_name="TEST",
        window_size=30,
        step=5
    )

    # ── Handle NaN/Inf ───────────────────────────────────────
    print("\nHandling NaN/Inf values...")

    # Drop columns that are entirely NaN
    train_features = train_features.dropna(axis=1, how="all")
    test_features = test_features.dropna(axis=1, how="all")

    # Keep only common columns between train and test
    # (excluding engine_id, cycle, target_RUL)
    train_feature_cols = [
        c for c in train_features.columns
        if c not in ["engine_id", "cycle", "target_RUL"]
    ]
    test_feature_cols = [
        c for c in test_features.columns
        if c not in ["engine_id", "cycle"]
    ]
    common_cols = list(set(train_feature_cols) & set(test_feature_cols))

    print(f"Common feature columns: {len(common_cols)}")

    # Apply common columns
    train_features = train_features[
        ["engine_id", "cycle", "target_RUL"] + common_cols
    ]
    test_features = test_features[
        ["engine_id", "cycle"] + common_cols
    ]

    # Replace inf with nan then fill with median
    train_features = train_features.replace([np.inf, -np.inf], np.nan)
    test_features = test_features.replace([np.inf, -np.inf], np.nan)

    # Fill NaN with column median from training data
    train_medians = train_features[common_cols].median()
    train_features[common_cols] = train_features[common_cols].fillna(train_medians)
    test_features[common_cols] = test_features[common_cols].fillna(train_medians)

    print(f"\n=== FINAL SHAPES ===")
    print(f"Train features : {train_features.shape}")
    print(f"Test features  : {test_features.shape}")
    print(f"Features count : {len(common_cols)}")

    # ── Log Metrics ──────────────────────────────────────────
    total_time = train_time + test_time
    mlflow.log_metric("train_extraction_time_seconds", train_time)
    mlflow.log_metric("test_extraction_time_seconds", test_time)
    mlflow.log_metric("total_extraction_time_seconds", total_time)
    mlflow.log_metric("n_extracted_features", len(common_cols))
    mlflow.log_param("feature_set", args.feature_set)
    mlflow.log_param("n_jobs", args.n_jobs)
    mlflow.log_param("window_size", 30)
    mlflow.log_param("step_size", 5)

    print(f"\n=== EXTRACTION METRICS ===")
    print(f"Total features : {len(common_cols)}")
    print(f"Total runtime  : {total_time:.2f}s")

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

    # Save metrics
    metrics = {
        "feature_set": args.feature_set,
        "window_size": 30,
        "step_size": 5,
        "n_extracted_features": len(common_cols),
        "train_rows": len(train_features),
        "test_rows": len(test_features),
        "train_extraction_time_seconds": round(train_time, 2),
        "test_extraction_time_seconds": round(test_time, 2),
        "total_extraction_time_seconds": round(total_time, 2),
        "train_rul_mean": round(float(train_features["target_RUL"].mean()), 2),
        "train_rul_std": round(float(train_features["target_RUL"].std()), 2)
    }

    with open(
        os.path.join(args.extraction_metrics, "metrics.json"), "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    print(f"\nFeature extraction complete!")
    print(f"Train : {args.train_features}")
    print(f"Test  : {args.test_features}")

    mlflow.end_run()

if __name__ == "__main__":
    main()
