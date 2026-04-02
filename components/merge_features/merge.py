import pandas as pd

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
    # Keep only labels early (BIG memory win)
    # -----------------------
    labels_df = raw_df[["asin", "reviewerID", "overall"]].copy()
    del raw_df  # free memory immediately

    # -----------------------
    # Merge ONLY keys first (reduce explosion risk)
    # -----------------------
    base = length_df[["asin", "reviewerID"]].copy()

    # align everything BEFORE merge
    length_df = length_df.drop(columns=["asin", "reviewerID"])
    sentiment_df = sentiment_df.drop(columns=["asin", "reviewerID"])
    tfidf_df = tfidf_df.drop(columns=["asin", "reviewerID"])
    sbert_df = sbert_df.drop(columns=["asin", "reviewerID"])

    # -----------------------
    # MEMORY-EFFICIENT CONCAT (NOT repeated merges)
    # -----------------------
    feature_df = pd.concat(
        [base, length_df, sentiment_df, tfidf_df, sbert_df],
        axis=1
    )

    # -----------------------
    # Attach labels last (small table)
    # -----------------------
    final_df = feature_df.merge(labels_df, on=["asin", "reviewerID"], how="inner")

    # -----------------------
    # Safety checks
    # -----------------------
    if final_df.empty:
        raise ValueError("Final merged dataframe is empty")

    if "overall" not in final_df.columns:
        raise ValueError("Missing label column")

    # -----------------------
    # Save output
    # -----------------------
    final_df.to_parquet(output_path, index=False)

    print("Merge successful")
    print("Final shape:", final_df.shape)