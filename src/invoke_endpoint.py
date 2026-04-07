import requests
import json
import pandas as pd

from azure.ai.ml import MLClient
from azure.identity import DefaultAzureCredential


# --------------------------------------------------
# Endpoint details
# --------------------------------------------------
ENDPOINT_URL = "https://amazon-review-endpoint-v2.qatarcentral.inference.ml.azure.com/score"
API_KEY = ""

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}


# --------------------------------------------------
# Connect to Azure ML
# --------------------------------------------------
def load_dataset():

    ml_client = MLClient(
        DefaultAzureCredential(),
        subscription_id="",
        resource_group="",
        workspace_name=""
    )

    dataset = ml_client.data.get(
        name="amazon_review_merged_features_deploy",
        version="1"
    )

    df = pd.read_parquet(dataset.path)
    return df


# --------------------------------------------------
# Feature builder (MATCH TRAINING)
# --------------------------------------------------
def build_features(df):

    feature_cols = [
        c for c in df.columns
        if any(prefix in c for prefix in [
            "tfidf_",
            "sbert_",
            "sentiment_",
            "length_",
            "review_length_"
        ])
    ]

    return df[feature_cols].fillna(0)


# --------------------------------------------------
# Main
# --------------------------------------------------
def main():

    df = load_dataset()

    X = build_features(df)

    payload = {
        "data": X.values.tolist()
    }

    response = requests.post(
        ENDPOINT_URL,
        headers=headers,
        json=payload
    )

    print("Response:")
    print(response.text)


if __name__ == "__main__":
    main()