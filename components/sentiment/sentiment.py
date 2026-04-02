import argparse
import pandas as pd
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

KEYS = ["asin", "reviewerID"]

def load_local_vader():
    """
    Load VADER lexicon from local file (NO internet)
    """
    local_path = os.path.join(os.path.dirname(__file__), "nltk_data")

    if not os.path.exists(local_path):
        raise RuntimeError(
            f"Local nltk_data folder not found at {local_path}"
        )

    # Tell nltk to look here
    nltk.data.path.append(local_path)

    try:
        nltk.data.find("sentiment/vader_lexicon.zip")
    except LookupError:
        raise RuntimeError(
            "vader_lexicon not found in local nltk_data folder. "
            "Make sure structure is: nltk_data/sentiment/vader_lexicon.zip"
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="reviewText")
    args = parser.parse_args()

    # -----------------------
    # Load local vader FIRST
    # -----------------------
    load_local_vader()

    sia = SentimentIntensityAnalyzer()

    # -----------------------
    # Load input parquet
    # -----------------------
    files = [f for f in os.listdir(args.input_path) if f.endswith(".parquet")]
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
    # Keep ONLY required columns
    # -----------------------
    out_df = df[KEYS].copy()

    out_df["sentiment_pos"] = scores.apply(lambda x: x["pos"]).values
    out_df["sentiment_neg"] = scores.apply(lambda x: x["neg"]).values
    out_df["sentiment_neu"] = scores.apply(lambda x: x["neu"]).values
    out_df["sentiment_compound"] = scores.apply(lambda x: x["compound"]).values

    # -----------------------
    # Save
    # -----------------------
    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, "data.parquet")

    out_df.to_parquet(output_file, index=False)

    print("Sentiment step completed successfully")
    print(f"Output shape: {out_df.shape}")


if __name__ == "__main__":
    main()