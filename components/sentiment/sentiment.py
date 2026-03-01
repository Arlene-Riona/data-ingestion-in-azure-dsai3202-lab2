import argparse
import pandas as pd
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer

# download vader lexicon if not present
nltk.download("vader_lexicon")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="reviewText")
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)

    sia = SentimentIntensityAnalyzer()

    # fill null safely
    texts = df[args.text_column].fillna("").astype(str)

    # compute sentiment scores
    scores = texts.apply(lambda x: sia.polarity_scores(x))

    df["sentiment_pos"] = scores.apply(lambda x: x["pos"])
    df["sentiment_neg"] = scores.apply(lambda x: x["neg"])
    df["sentiment_neu"] = scores.apply(lambda x: x["neu"])
    df["sentiment_compound"] = scores.apply(lambda x: x["compound"])

    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()