import pandas as pd
import argparse
import os

def safe_load(path):
    df = pd.read_parquet(path)
    return df.reset_index(drop=True)

def main(length_path, sentiment_path, tfidf_path, sbert_path, raw_path, output_path):

    # -----------------------
    # Load datasets
    # -----------------------
    length_df = safe_load(length_path)
    sentiment_df = safe_load(sentiment_path)
    tfidf_df = safe_load(tfidf_path)
    sbert_df = safe_load(sbert_path)
    raw_df = safe_load(raw_path)

    # -----------------------
    # Labels
    # -----------------------
    labels_df = raw_df[["asin", "reviewerID", "overall"]].copy()
    del raw_df

    # -----------------------
    # Base keys
    # -----------------------
    base = length_df[["asin", "reviewerID"]].copy()

    length_df = length_df.drop(columns=["asin", "reviewerID"])
    sentiment_df = sentiment_df.drop(columns=["asin", "reviewerID"])
    tfidf_df = tfidf_df.drop(columns=["asin", "reviewerID"])
    sbert_df = sbert_df.drop(columns=["asin", "reviewerID"])

    # -----------------------
    # Merge features
    # -----------------------
    feature_df = pd.concat(
        [base, length_df, sentiment_df, tfidf_df, sbert_df],
        axis=1
    )

    final_df = feature_df.merge(labels_df, on=["asin", "reviewerID"], how="inner")

    if final_df.empty:
        raise ValueError("Final merged dataframe is empty")

    if "overall" not in final_df.columns:
        raise ValueError("Missing label column")

    # -----------------------
    # IMPORTANT: ensure output folder exists
    # -----------------------
    os.makedirs(output_path, exist_ok=True)

    output_file = os.path.join(output_path, "merged.parquet")

    final_df.to_parquet(output_file, index=False)

    print("Merge successful")
    print("Final shape:", final_df.shape)
    print("Saved to:", output_file)


# -----------------------
# AzureML ENTRY POINT
# -----------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--length", required=True)
    parser.add_argument("--sentiment", required=True)
    parser.add_argument("--tfidf", required=True)
    parser.add_argument("--sbert", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output_path", required=True)

    args = parser.parse_args()

    main(
        args.length,
        args.sentiment,
        args.tfidf,
        args.sbert,
        args.raw,
        args.output_path
    )