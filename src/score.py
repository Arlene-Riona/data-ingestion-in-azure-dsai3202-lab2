import json
import os
import joblib
import pandas as pd

model = None

# SAME feature logic as training
def build_features(df):
    feature_cols = [
        c for c in df.columns
        if any(prefix in c for prefix in [
            "tfidf_", "sbert_", "sentiment_", "length_", "review_length_"
        ])
    ]

    if len(feature_cols) == 0:
        raise RuntimeError("No feature columns found")

    return df[feature_cols].fillna(0)


def init():
    global model
    model_path = os.path.join(os.getenv("AZUREML_MODEL_DIR"), "model.pkl")
    model = joblib.load(model_path)


def run(raw_data):
    try:
        data = json.loads(raw_data)

        # Expecting list of dicts (records)
        df = pd.DataFrame(data["data"])

        X = build_features(df)

        preds = model.predict(X)

        return {"predictions": preds.tolist()}

    except Exception as e:
        return {"error": str(e)}