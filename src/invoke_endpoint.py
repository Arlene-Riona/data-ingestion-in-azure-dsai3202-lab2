import requests
import json
import pandas as pd

ENDPOINT_URL = "<your-endpoint-url>"
API_KEY = "<your-api-key>"

headers = {
    "Content-Type": "application/json",
    "Authorization": f"Bearer {API_KEY}"
}

def main():
    df = pd.read_parquet("sample.parquet")  # small test sample

    X = df.select_dtypes(include=["number"]).fillna(0)

    payload = {
        "data": X.values.tolist()
    }

    response = requests.post(
        ENDPOINT_URL,
        headers=headers,
        data=json.dumps(payload)
    )

    print(response.text)

if __name__ == "__main__":
    main()