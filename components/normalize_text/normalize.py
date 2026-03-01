import re
import argparse
import pandas as pd


def normalize_text(text: str) -> str:
    if pd.isna(text):
        return ""

    # lowercase
    text = text.lower()

    # replace URLs
    text = re.sub(r"http\S+|www\S+", " URL ", text)

    # replace numbers
    text = re.sub(r"\d+", " NUMBER ", text)

    # remove punctuation
    text = re.sub(r"[^\w\s]", " ", text)

    # remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="reviewText")
    args = parser.parse_args()

    # read data (expects CSV)
    df = pd.read_csv(args.input_path)

    # normalize text column
    df[args.text_column] = df[args.text_column].apply(normalize_text)

    # remove empty or very short reviews (<10 characters)
    df = df[df[args.text_column].str.len() >= 10]

    # save output
    df.to_csv(args.output_path, index=False)


if __name__ == "__main__":
    main()