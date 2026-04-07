import argparse
import os
import time
import mlflow
import joblib
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score


# -------------------------
# Args
# -------------------------
def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data", type=str, required=True)
    parser.add_argument("--val_data", type=str, required=True)
    parser.add_argument("--test_data", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--C", type=float, default=0.3213738590686788)
    parser.add_argument("--max_iter", type=int, default=100)
    return parser.parse_args()


# -------------------------
# Load
# -------------------------
def load_data(path):
    files = [f for f in os.listdir(path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet found in {path}")

    return pd.read_parquet(os.path.join(path, files[0]))


# -------------------------
# Labels
# -------------------------
def create_labels(df):
    if "overall" not in df.columns:
        raise RuntimeError("Missing label column 'overall'")

    df["label"] = (df["overall"] >= 4).astype(int)
    return df

# -------------------------
# Features (SAFE VERSION)
# -------------------------
def build_features(df):

    feature_cols = [
        c for c in df.columns
        if "sbert_" in c
    ]

    if len(feature_cols) == 0:
        raise RuntimeError(f"No SBERT features found. Columns: {list(df.columns)}")

    return df[feature_cols].fillna(0)


# -------------------------
# Eval
# -------------------------
def evaluate(model, X, y, split):
    preds = model.predict(X)
    probs = model.predict_proba(X)[:, 1]

    acc = accuracy_score(y, preds)
    prec = precision_score(y, preds)
    rec = recall_score(y, preds)
    f1 = f1_score(y, preds)
    auc = roc_auc_score(y, probs)

    mlflow.log_metric(f"{split}_accuracy", acc)
    mlflow.log_metric(f"{split}_precision", prec)
    mlflow.log_metric(f"{split}_recall", rec)
    mlflow.log_metric(f"{split}_f1", f1)
    mlflow.log_metric(f"{split}_auc", auc)

    print(f"{split}: acc={acc:.4f}, f1={f1:.4f}")


# -------------------------
# Main
# -------------------------
def main():
    args = parse_args()
    start = time.time()

    print("Loading data...")
    train_df = load_data(args.train_data)
    val_df = load_data(args.val_data)
    test_df = load_data(args.test_data)

    print("Creating labels...")
    train_df = create_labels(train_df)
    val_df = create_labels(val_df)
    test_df = create_labels(test_df)

    print("Building features...")
    X_train = build_features(train_df)
    y_train = train_df["label"]

    X_val = build_features(val_df)
    y_val = val_df["label"]

    X_test = build_features(test_df)
    y_test = test_df["label"]
    mlflow.log_param("C", args.C)
    mlflow.log_param("max_iter", args.max_iter)

    print("Training model...")
    model = LogisticRegression(
        C=args.C,
        max_iter=args.max_iter,
        solver="liblinear")
    model.fit(X_train, y_train)

    print("Evaluating...")
    evaluate(model, X_train, y_train, "train")
    evaluate(model, X_val, y_val, "val")
    evaluate(model, X_test, y_test, "test")

    print("Saving model...")
    os.makedirs(args.output, exist_ok=True)

    model_path = os.path.join(args.output, "model.pkl")
    joblib.dump(model, model_path)

    mlflow.log_artifact(model_path)
    mlflow.log_metric("runtime_seconds", time.time() - start)

    print("Done.")


if __name__ == "__main__":
    main()