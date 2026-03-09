# ============================================================
# Split Dataset Component
# Splits GA-selected features into train and validation sets
# ============================================================

import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import mlflow

def parse_args():
    parser = argparse.ArgumentParser("split_dataset")
    parser.add_argument("--train_ga_selected", type=str)
    parser.add_argument("--test_size", type=float, default=0.2)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--train_split", type=str)
    parser.add_argument("--val_split", type=str)
    parser.add_argument("--split_metrics", type=str)
    return parser.parse_args()

def load_parquet_folder(path):
    """Load all parquet files from a folder"""
    files = [f for f in os.listdir(path) if f.endswith(".parquet")]
    dfs = [pd.read_parquet(os.path.join(path, f)) for f in files]
    return pd.concat(dfs, ignore_index=True)

def main():
    args = parse_args()

    mlflow.start_run()

    print("=" * 50)
    print("SPLIT DATASET COMPONENT")
    print("=" * 50)

    # ── Load Data ────────────────────────────────────────────
    print("\nLoading GA selected features...")
    df = load_parquet_folder(args.train_ga_selected)

    print(f"Loaded shape : {df.shape}")

    # Separate features and target
    target_col = "target_RUL"
    feature_cols = [c for c in df.columns if c != target_col]

    X = df[feature_cols]
    y = df[target_col]

    print(f"Features     : {len(feature_cols)}")
    print(f"Samples      : {len(df)}")

    # ── Split ────────────────────────────────────────────────
    print(f"\n=== SPLITTING DATA ===")
    print(f"Test size    : {args.test_size}")
    print(f"Random seed  : {args.random_seed}")

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=args.test_size,
        random_state=args.random_seed
    )

    # Reconstruct dataframes
    train_df = X_train.copy()
    train_df[target_col] = y_train.values

    val_df = X_val.copy()
    val_df[target_col] = y_val.values

    print(f"\n=== SPLIT SUMMARY ===")
    print(f"Train samples : {len(train_df)}")
    print(f"Val samples   : {len(val_df)}")
    print(f"Train RUL mean: {y_train.mean():.2f}")
    print(f"Val RUL mean  : {y_val.mean():.2f}")

    # ── Log Metrics ──────────────────────────────────────────
    mlflow.log_metric("train_samples", len(train_df))
    mlflow.log_metric("val_samples", len(val_df))
    mlflow.log_metric("train_rul_mean", y_train.mean())
    mlflow.log_metric("val_rul_mean", y_val.mean())
    mlflow.log_param("test_size", args.test_size)
    mlflow.log_param("random_seed", args.random_seed)

    # ── Save Outputs ─────────────────────────────────────────
    os.makedirs(args.train_split, exist_ok=True)
    os.makedirs(args.val_split, exist_ok=True)
    os.makedirs(args.split_metrics, exist_ok=True)

    train_df.to_parquet(
        os.path.join(args.train_split, "train_split.parquet")
    )
    val_df.to_parquet(
        os.path.join(args.val_split, "val_split.parquet")
    )

    # Save split metrics
    metrics = {
        "train_samples": len(train_df),
        "val_samples": len(val_df),
        "total_samples": len(df),
        "test_size": args.test_size,
        "random_seed": args.random_seed,
        "n_features": len(feature_cols),
        "train_rul_mean": round(y_train.mean(), 2),
        "val_rul_mean": round(y_val.mean(), 2),
        "train_rul_std": round(y_train.std(), 2),
        "val_rul_std": round(y_val.std(), 2)
    }

    with open(os.path.join(args.split_metrics, "split_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nSplit dataset component complete!")
    print(f"Train split  : {args.train_split}")
    print(f"Val split    : {args.val_split}")
    print(f"Metrics      : {args.split_metrics}")

    mlflow.end_run()

if __name__ == "__main__":
    main()