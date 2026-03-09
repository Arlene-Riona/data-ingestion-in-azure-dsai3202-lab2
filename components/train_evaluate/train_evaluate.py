# ============================================================
# Train and Evaluate Component
# Trains regression model and evaluates RUL predictions
# ============================================================

import argparse
import os
import json
import time
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib
import mlflow

def parse_args():
    parser = argparse.ArgumentParser("train_evaluate")
    parser.add_argument("--train_split", type=str)
    parser.add_argument("--val_split", type=str)
    parser.add_argument("--test_ga_selected", type=str)
    parser.add_argument("--model_type", type=str, default="RandomForest")
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--max_depth", type=int, default=10)
    parser.add_argument("--random_seed", type=int, default=42)
    parser.add_argument("--model_output", type=str)
    parser.add_argument("--evaluation_metrics", type=str)
    parser.add_argument("--test_predictions", type=str)
    return parser.parse_args()

def load_parquet_folder(path):
    """Load all parquet files from a folder"""
    files = [f for f in os.listdir(path) if f.endswith(".parquet")]
    dfs = [pd.read_parquet(os.path.join(path, f)) for f in files]
    return pd.concat(dfs, ignore_index=True)

def get_model(model_type, n_estimators, max_depth, random_seed):
    """Return model based on type"""
    print(f"\nInitializing model: {model_type}")

    if model_type == "RandomForest":
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_seed,
            n_jobs=-1
        )
    elif model_type == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_seed,
            n_jobs=-1
        )
    elif model_type == "LightGBM":
        from lightgbm import LGBMRegressor
        return LGBMRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_seed,
            n_jobs=-1
        )
    else:
        print(f"Unknown model type {model_type}, defaulting to RandomForest")
        return RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_seed,
            n_jobs=-1
        )

def evaluate_predictions(y_true, y_pred, split_name):
    """Calculate and print evaluation metrics"""
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)

    print(f"\n=== {split_name} METRICS ===")
    print(f"RMSE : {rmse:.4f}")
    print(f"MAE  : {mae:.4f}")
    print(f"R2   : {r2:.4f}")

    return rmse, mae, r2

def main():
    args = parse_args()

    mlflow.start_run()

    print("=" * 50)
    print("TRAIN AND EVALUATE COMPONENT")
    print("=" * 50)

    # ── Load Data ────────────────────────────────────────────
    print("\nLoading data splits...")
    train_df = load_parquet_folder(args.train_split)
    val_df = load_parquet_folder(args.val_split)
    test_df = load_parquet_folder(args.test_ga_selected)

    print(f"Train split : {train_df.shape}")
    print(f"Val split   : {val_df.shape}")
    print(f"Test data   : {test_df.shape}")

    # Separate features and target
    target_col = "target_RUL"
    feature_cols = [c for c in train_df.columns if c != target_col]

    X_train = train_df[feature_cols].values
    y_train = train_df[target_col].values
    X_val = val_df[feature_cols].values
    y_val = val_df[target_col].values
    X_test = test_df[feature_cols].values

    print(f"\nFeatures    : {len(feature_cols)}")
    print(f"Train size  : {len(X_train)}")
    print(f"Val size    : {len(X_val)}")
    print(f"Test size   : {len(X_test)}")

    # ── Train Model ──────────────────────────────────────────
    model = get_model(
        args.model_type,
        args.n_estimators,
        args.max_depth,
        args.random_seed
    )

    print(f"\n=== TRAINING ===")
    start_time = time.time()
    model.fit(X_train, y_train)
    train_time = time.time() - start_time
    print(f"Training time : {train_time:.2f} seconds")

    # ── Evaluate on Validation Set ───────────────────────────
    val_preds = model.predict(X_val)
    val_rmse, val_mae, val_r2 = evaluate_predictions(
        y_val, val_preds, "VALIDATION"
    )

    # ── Evaluate on Training Set ─────────────────────────────
    train_preds = model.predict(X_train)
    train_rmse, train_mae, train_r2 = evaluate_predictions(
        y_train, train_preds, "TRAINING"
    )

    # ── Predict on Test Set ──────────────────────────────────
    print(f"\n=== PREDICTING ON TEST SET ===")
    test_preds = model.predict(X_test)
    print(f"Test predictions shape : {test_preds.shape}")
    print(f"Test RUL mean          : {test_preds.mean():.2f}")
    print(f"Test RUL min           : {test_preds.min():.2f}")
    print(f"Test RUL max           : {test_preds.max():.2f}")

    # ── Feature Importance ───────────────────────────────────
    if hasattr(model, "feature_importances_"):
        importance_df = pd.DataFrame({
            "feature": feature_cols,
            "importance": model.feature_importances_
        }).sort_values("importance", ascending=False)

        print(f"\n=== TOP 10 MOST IMPORTANT FEATURES ===")
        print(importance_df.head(10).to_string())

    # ── Log Metrics to MLflow ────────────────────────────────
    mlflow.log_metric("val_rmse", val_rmse)
    mlflow.log_metric("val_mae", val_mae)
    mlflow.log_metric("val_r2", val_r2)
    mlflow.log_metric("train_rmse", train_rmse)
    mlflow.log_metric("train_mae", train_mae)
    mlflow.log_metric("train_r2", train_r2)
    mlflow.log_metric("training_time_seconds", train_time)
    mlflow.log_param("model_type", args.model_type)
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)
    mlflow.log_param("n_features", len(feature_cols))

    # ── Save Outputs ─────────────────────────────────────────
    os.makedirs(args.model_output, exist_ok=True)
    os.makedirs(args.evaluation_metrics, exist_ok=True)
    os.makedirs(args.test_predictions, exist_ok=True)

    # Save model
    joblib.dump(
        model,
        os.path.join(args.model_output, "model.pkl")
    )

    # Save evaluation metrics
    metrics = {
        "model_type": args.model_type,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "n_features": len(feature_cols),
        "training_time_seconds": round(train_time, 2),
        "validation": {
            "rmse": round(val_rmse, 4),
            "mae": round(val_mae, 4),
            "r2": round(val_r2, 4)
        },
        "training": {
            "rmse": round(train_rmse, 4),
            "mae": round(train_mae, 4),
            "r2": round(train_r2, 4)
        }
    }

    with open(
        os.path.join(args.evaluation_metrics, "evaluation_metrics.json"), "w"
    ) as f:
        json.dump(metrics, f, indent=2)

    # Save test predictions
    predictions_df = pd.DataFrame({
        "predicted_RUL": test_preds
    })
    predictions_df.to_parquet(
        os.path.join(args.test_predictions, "test_predictions.parquet")
    )
    predictions_df.to_csv(
        os.path.join(args.test_predictions, "test_predictions.csv"),
        index=False
    )

    print(f"\n=== FINAL SUMMARY ===")
    print(f"Model         : {args.model_type}")
    print(f"Val RMSE      : {val_rmse:.4f}")
    print(f"Val MAE       : {val_mae:.4f}")
    print(f"Val R2        : {val_r2:.4f}")
    print(f"Training time : {train_time:.2f}s")

    print(f"\nTrain and evaluate component complete!")

    mlflow.end_run()

if __name__ == "__main__":
    main()
