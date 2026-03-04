import re
import argparse
import pandas as pd
import os


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
    # These argument names match what component.yml actually passes (--data and --out)
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="reviewText")
    args = parser.parse_args()

    # Find the parquet file inside the input folder
    files = [f for f in os.listdir(args.data) if f.endswith('.parquet')]
    if not files:
        raise FileNotFoundError(f"No parquet file found in {args.data}")

    input_full_path = os.path.join(args.data, files[0])
    print(f"Reading from: {input_full_path}")

    # Read the parquet file (NOT csv)
    df = pd.read_parquet(input_full_path)

    # Normalize text column
    df[args.text_column] = df[args.text_column].apply(normalize_text)

    # Remove empty or very short reviews (<10 characters)
    df = df[df[args.text_column].str.len() >= 10]

    # Create the output directory if it doesn't exist
    os.makedirs(args.out, exist_ok=True)

    # Save as data.parquet so downstream components can find it
    output_path = os.path.join(args.out, "data.parquet")
    df.to_parquet(output_path, index=False)
    print(f"Saved normalized data to: {output_path}")
    print(f"Rows after normalization: {len(df)}")


if __name__ == "__main__":
    main()