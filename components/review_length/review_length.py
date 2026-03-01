import argparse
import pandas as pd


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="reviewText")
    args = parser.parse_args()

    df = pd.read_csv(args.input_path)

    # safe text
    text = df[args.text_column].fillna("").astype(str)

    # features
    df["review_len_chars"] = text.str.len()
    df["review_len_words"] = text.str.split().str.len()

    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()