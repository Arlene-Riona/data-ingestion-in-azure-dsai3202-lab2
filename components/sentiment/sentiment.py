import argparse
import pandas as pd
import os
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

KEYS = ["asin", "reviewerID"]


def setup_nltk():
    # Force NLTK to look in current folder
    nltk.data.path.append(os.getcwd())

    lexicon_path = os.path.join(os.getcwd(), "vader_lexicon.zip")

    if not os.path.exists(lexicon_path):
        raise RuntimeError(
            "vader_lexicon.zip not found in working directory. "
            "You MUST include it with your code."
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="reviewText")
    args = parser.parse_args()

    setup_nltk()

    files = [f for f in os.listdir(args.input_path) if f.endswith('.parquet')]
    if not files:
        raise FileNotFoundError(f"No parquet file found in {args.input_path}")

    input_file = os.path.join(args.input_path, files[0])
    df = pd.read_parquet(input_file)

    sia = SentimentIntensityAnalyzer()

    texts = df[args.text_column].fillna("").astype(str)
    scores = texts.apply(lambda x: sia.polarity_scores(x))

    out_df = df[KEYS].copy()
    out_df["sentiment_pos"] = scores.apply(lambda x: x["pos"]).values
    out_df["sentiment_neg"] = scores.apply(lambda x: x["neg"]).values
    out_df["sentiment_neu"] = scores.apply(lambda x: x["neu"]).values
    out_df["sentiment_compound"] = scores.apply(lambda x: x["compound"]).values

    os.makedirs(args.output_path, exist_ok=True)
    output_file = os.path.join(args.output_path, "data.parquet")
    out_df.to_parquet(output_file, index=False)

    print(f"Sentiment features saved to {output_file}")
    print(f"Output shape: {out_df.shape}")


if __name__ == "__main__":
    main()