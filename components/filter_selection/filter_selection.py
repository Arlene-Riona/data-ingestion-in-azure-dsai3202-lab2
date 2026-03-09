# ============================================================
# Filter-Based Feature Selection Component
# Variance threshold, correlation filtering, mutual information
# ============================================================

import argparse
import os
import json
import pandas as pd
import numpy as np
from sklearn.feature_selection import VarianceThreshold, mutual_info_regression
import mlflow

def parse_args():
    parser = argparse.ArgumentParser("filter_selection")
    parser.add_argument("--train_features", type=str)
    parser.add_argument("--test_features", type=str)
    parser.add_argument("--variance_threshold", type=float, default=0.01)
    parser.add_argument("--correlation_threshold", type=float, default=0.95)
    parser.add_argument("--mutual_info_percentile", type=int, default=50)
    parser.add_argument("--train_filtered", type=str)
    parser.add_argument("--test_filtered", type=str)
    parser.add_argument("--selected_features", type=str)
    parser.add_argument("--filter_metrics", type=str)
    return parser.parse_args()

def load_parquet_folder(path):
    """Load all parquet files from a folder"""
    files = [f for f in os.listdir(path) if f.endswith(".parquet")]
    dfs = [pd.read_parquet(os.path.join(path, f)) for f in files]
    return pd.concat(dfs, ignore_index=True)

def apply_variance_threshold(train_df, test_df, feature_cols, threshold):
    """Remove features with variance below threshold"""
    print(f"\n=== STEP 1: Variance Threshold (threshold={threshold}) ===")
    print(f"Features before : {len(feature_cols)}")

    selector = VarianceThreshold(threshold=threshold)
    selector.fit(train_df[feature_cols])

    selected = [feature_cols[i] for i, s in
                enumerate(selector.get_support()) if s]

    removed = len(feature_cols) - len(selected)
    print(f"Features removed : {removed}")
    print(f"Features after   : {len(selected)}")

    return selected

def apply_correlation_filter(train_df, feature_cols, threshold):
    """Remove highly correlated features"""
    print(f"\n=== STEP 2: Correlation Filter (threshold={threshold}) ===")
    print(f"Features before : {len(feature_cols)}")

    corr_matrix = train_df[feature_cols].corr().abs()

    # Upper triangle of correlation matrix
    upper = corr_matrix.where(
        np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
    )

    # Find features with correlation above threshold
    to_drop = [col for col in upper.columns
               if any(upper[col] > threshold)]

    selected = [f for f in feature_cols if f not in to_drop]

    print(f"Features removed : {len(to_drop)}")
    print(f"Features after   : {len(selected)}")

    return selected

def apply_mutual_information(train_df, feature_cols, target_col, percentile):
    """Keep top N% features by mutual information score"""
    print(f"\n=== STEP 3: Mutual Information (top {percentile}%) ===")
    print(f"Features before : {len(feature_cols)}")

    X = train_df[feature_cols].values
    y = train_df[target_col].values

    # Compute mutual information scores
    mi_scores = mutual_info_regression(X, y, random_state=42)
    mi_series = pd.Series(mi_scores, index=feature_cols)

    # Keep top percentile
    threshold = np.percentile(mi_scores, 100 - percentile)
    selected = mi_series[mi_series >= threshold].index.tolist()

    print(f"Features removed : {len(feature_cols) - len(selected)}")
    print(f"Features after   : {len(selected)}")

    # Show top 10 most informative features
    print("\nTop 10 most informative features:")
    print(mi_series.nlargest(10))

    return selected, mi_series.to_dict()

def main():
    args = parse_args()

    mlflow.start_run()

    print("=" * 50)
    print("FILTER-BASED FEATURE SELECTION COMPONENT")
    print("=" * 50)

    # ── Load Data ────────────────────────────────────────────
    print("\nLoading extracted features...")
    train_df = load_parquet_folder(args.train_features)
    test_df = load_parquet_folder(args.test_features)

    print(f"Train loaded : {train_df.shape}")
    print(f"Test loaded  : {test_df.shape}")

    # Separate features from target
    target_col = "target_RUL"
    feature_cols = [c for c in train_df.columns if c != target_col]
    initial_count = len(feature_cols)

    print(f"\nInitial features : {initial_count}")

    # ── Step 1: Variance Threshold ───────────────────────────
    feature_cols = apply_variance_threshold(
        train_df, test_df,
        feature_cols,
        args.variance_threshold
    )
    after_variance = len(feature_cols)

    # ── Step 2: Correlation Filter ───────────────────────────
    feature_cols = apply_correlation_filter(
        train_df,
        feature_cols,
        args.correlation_threshold
    )
    after_correlation = len(feature_cols)

    # ── Step 3: Mutual Information ───────────────────────────
    feature_cols, mi_scores = apply_mutual_information(
        train_df,
        feature_cols,
        target_col,
        args.mutual_info_percentile
    )
    after_mi = len(feature_cols)

    # ── Summary ──────────────────────────────────────────────
    print(f"\n=== FILTER SELECTION SUMMARY ===")
    print(f"Initial features       : {initial_count}")
    print(f"After variance filter  : {after_variance}")
    print(f"After correlation      : {after_correlation}")
    print(f"After mutual info      : {after_mi}")
    print(f"Total removed          : {initial_count - after_mi}")
    print(f"Features remaining     : {after_mi}")

    # ── Apply Selection to DataFrames ────────────────────────
    train_filtered = train_df[feature_cols + [target_col]]
    test_filtered = test_df[feature_cols]

    # ── Log Metrics ──────────────────────────────────────────
    mlflow.log_metric("initial_features", initial_count)
    mlflow.log_metric("after_variance_filter", after_variance)
    mlflow.log_metric("after_correlation_filter", after_correlation)
    mlflow.log_metric("after_mutual_info_filter", after_mi)
    mlflow.log_metric("total_features_removed", initial_count - after_mi)
    mlflow.log_param("variance_threshold", args.variance_threshold)
    mlflow.log_param("correlation_threshold", args.correlation_threshold)
    mlflow.log_param("mutual_info_percentile", args.mutual_info_percentile)

    # ── Save Outputs ─────────────────────────────────────────
    os.makedirs(args.train_filtered, exist_ok=True)
    os.makedirs(args.test_filtered, exist_ok=True)
    os.makedirs(args.selected_features, exist_ok=True)
    os.makedirs(args.filter_metrics, exist_ok=True)

    train_filtered.to_parquet(
        os.path.join(args.train_filtered, "train_filtered.parquet")
    )
    test_filtered.to_parquet(
        os.path.join(args.test_filtered, "test_filtered.parquet")
    )

    # Save selected feature names
    with open(os.path.join(args.selected_features, "selected_features.json"), "w") as f:
        json.dump({"selected_features": feature_cols}, f, indent=2)

    # Save filter metrics
    metrics = {
        "initial_features": initial_count,
        "after_variance_filter": after_variance,
        "after_correlation_filter": after_correlation,
        "after_mutual_info_filter": after_mi,
        "total_removed": initial_count - after_mi,
        "final_feature_count": after_mi,
        "variance_threshold": args.variance_threshold,
        "correlation_threshold": args.correlation_threshold,
        "mutual_info_percentile": args.mutual_info_percentile
    }

    with open(os.path.join(args.filter_metrics, "filter_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"\nFilter selection complete!")
    print(f"Train filtered : {args.train_filtered}")
    print(f"Test filtered  : {args.test_filtered}")

    mlflow.end_run()

if __name__ == "__main__":
    main()
