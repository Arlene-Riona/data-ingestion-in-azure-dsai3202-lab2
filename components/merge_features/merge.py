import argparse
import os
import pandas as pd

KEYS = ["asin", "reviewerID"]

def read_df(folder_path: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(folder_path, "data.parquet"))

def write_df(df: pd.DataFrame, out_folder: str):
    os.makedirs(out_folder, exist_ok=True)
    df.to_parquet(os.path.join(out_folder, "data.parquet"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)
    parser.add_argument("--tfidf", type=str, required=True)
    parser.add_argument("--sbert", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    length_df = read_df(args.length)
    sentiment_df = read_df(args.sentiment)
    tfidf_df = read_df(args.tfidf)
    sbert_df = read_df(args.sbert)

    # Merge all feature sets on entity keys
    merged = (
        length_df
        .merge(sentiment_df, on=KEYS, how="inner")
        .merge(tfidf_df, on=KEYS, how="inner")
        .merge(sbert_df, on=KEYS, how="inner")
    )

    write_df(merged, args.out)

    print("Final merged dataset shape:", merged.shape)

if __name__ == "__main__":
    main()