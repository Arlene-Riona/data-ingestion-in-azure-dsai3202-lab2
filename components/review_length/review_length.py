import argparse
import os
from pathlib import Path
import pandas as pd

KEYS = ["asin", "reviewerID"]


def _find_first_data_file(folder: Path) -> Path:
    candidates = []
    for ext in ("*.parquet", "*.csv", "*.tsv"):
        candidates.extend(sorted(folder.rglob(ext)))
    if not candidates:
        raise FileNotFoundError(
            f"No data file found in: {folder}. Expected .parquet/.csv/.tsv inside the folder."
        )
    return candidates[0]


def _read_df(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    if path.suffix.lower() == ".tsv":
        return pd.read_csv(path, sep="\t")
    raise ValueError(f"Unsupported file type: {path}")


def main(input_path: str, output_path: str):
    in_dir = Path(input_path)
    out_dir = Path(output_path)
    out_dir.mkdir(parents=True, exist_ok=True)

    data_file = _find_first_data_file(in_dir)
    df = _read_df(data_file)

    # Try common text column names
    text_col = None
    for c in ["reviewText", "review_text", "text", "review", "summary"]:
        if c in df.columns:
            text_col = c
            break
    if text_col is None:
        raise ValueError(f"Could not find a review text column. Columns: {list(df.columns)}")

    # Compute length features
    s = df[text_col].fillna("").astype(str)

    # Only keep keys + new features (NOT the full dataframe)
    # This prevents a memory explosion when merging with other feature outputs
    out_df = df[KEYS].copy()
    out_df["review_length_chars"] = s.str.len().values
    out_df["review_length_words"] = s.str.split().str.len().values

    # Save as parquet inside output folder
    out_file = out_dir / "data.parquet"
    out_df.to_parquet(out_file, index=False)

    print(f"Read from:  {data_file}")
    print(f"Wrote to:   {out_file}")
    print(f"Output shape: {out_df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    args = parser.parse_args()
    main(args.input_path, args.output_path)