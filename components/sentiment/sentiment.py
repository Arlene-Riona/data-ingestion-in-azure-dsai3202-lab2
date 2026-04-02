"""
import argparse
import pandas as pd
import nltk
import os
from nltk.sentiment import SentimentIntensityAnalyzer

nltk.download("vader_lexicon")

KEYS = ["asin", "reviewerID"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="reviewText")
    args = parser.parse_args()

    # Find the parquet file inside the folder
    files = [f for f in os.listdir(args.input_path) if f.endswith('.parquet')]
    if not files:
        raise FileNotFoundError(f"No parquet file found in {args.input_path}")

    input_file = os.path.join(args.input_path, files[0])
    df = pd.read_parquet(input_file)

    sia = SentimentIntensityAnalyzer()

    texts = df[args.text_column].fillna("").astype(str)
    scores = texts.apply(lambda x: sia.polarity_scores(x))

    # Only keep keys + sentiment features (NOT the full dataframe)
    # Saving the full dataframe causes a memory explosion during merge
    out_df = df[KEYS].copy()
    out_df["sentiment_pos"]      = scores.apply(lambda x: x["pos"]).values
    out_df["sentiment_neg"]      = scores.apply(lambda x: x["neg"]).values
    out_df["sentiment_neu"]      = scores.apply(lambda x: x["neu"]).values
    out_df["sentiment_compound"] = scores.apply(lambda x: x["compound"]).values

    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, "data.parquet")
    out_df.to_parquet(output_file, index=False)

    print(f"Sentiment features saved to {output_file}")
    print(f"Output shape: {out_df.shape}")

if __name__ == "__main__":
    main()

"""
import argparse
import pandas as pd
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# -----------------------
# SAFE VADER LOADING (NO RUNTIME DOWNLOAD)
# -----------------------
try:
    sia = SentimentIntensityAnalyzer()
except LookupError:
    raise RuntimeError(
        "VADER lexicon not found in environment. "
        "Do NOT download at runtime. Ensure it exists in the Azure ML environment."
    )

KEYS = ["asin", "reviewerID"]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="reviewText")
    args = parser.parse_args()

    # -----------------------
    # Load input data
    # -----------------------
    files = [f for f in os.listdir(args.input_path) if f.endswith('.parquet')]
    if not files:
        raise FileNotFoundError(f"No parquet file found in {args.input_path}")

    input_file = os.path.join(args.input_path, files[0])
    df = pd.read_parquet(input_file)

    # -----------------------
    # Compute sentiment
    # -----------------------
    texts = df[args.text_column].fillna("").astype(str)
    scores = texts.apply(lambda x: sia.polarity_scores(x))

    # -----------------------
    # Keep only required columns
    # -----------------------
    out_df = df[KEYS].copy()
    out_df["sentiment_pos"]      = scores.apply(lambda x: x["pos"]).values
    out_df["sentiment_neg"]      = scores.apply(lambda x: x["neg"]).values
    out_df["sentiment_neu"]      = scores.apply(lambda x: x["neu"]).values
    out_df["sentiment_compound"] = scores.apply(lambda x: x["compound"]).values

    # -----------------------
    # Save output
    # -----------------------
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, "data.parquet")
    out_df.to_parquet(output_file, index=False)

    print(f"Sentiment features saved to {output_file}")
    print(f"Output shape: {out_df.shape}")

if __name__ == "__main__":
    main()