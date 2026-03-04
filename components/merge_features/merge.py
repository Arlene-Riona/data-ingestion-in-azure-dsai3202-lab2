import argparse
import os
import pandas as pd

KEYS = ["asin", "reviewerID"]
CHUNK_SIZE = 50_000


def read_first_parquet(folder_path: str) -> pd.DataFrame:
    files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet files found in {folder_path}")
    file_path = os.path.join(folder_path, files[0])
    print(f"Reading: {file_path}")
    return pd.read_parquet(file_path)


def downcast(df: pd.DataFrame) -> pd.DataFrame:
    for col in df.select_dtypes(include=["float64"]).columns:
        df[col] = df[col].astype("float32")
    for col in df.select_dtypes(include=["int64"]).columns:
        df[col] = df[col].astype("int32")
    return df


def make_unique(df: pd.DataFrame, name: str) -> pd.DataFrame:
    """
    Ensure KEYS uniquely identify rows.
    If duplicates exist, keep the last occurrence.
    """
    before = len(df)
    dup_count = df.duplicated(subset=KEYS).sum()
    if dup_count > 0:
        print(f"{name}: found {dup_count} duplicate keys. Deduplicating (keep='last')...")
        df = df.drop_duplicates(subset=KEYS, keep="last")
    after = len(df)
    print(f"{name}: rows {before} -> {after}")
    return df


def prep(df: pd.DataFrame, name: str) -> pd.DataFrame:
    df = downcast(df)
    df = make_unique(df, name)
    return df.set_index(KEYS).sort_index()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--length", type=str, required=True)
    parser.add_argument("--sentiment", type=str, required=True)
    parser.add_argument("--tfidf", type=str, required=True)
    parser.add_argument("--sbert", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)

    print("Loading + preparing dataframes...")
    length_df = prep(read_first_parquet(args.length), "length")
    sentiment_df = prep(read_first_parquet(args.sentiment), "sentiment")
    tfidf_df = prep(read_first_parquet(args.tfidf), "tfidf")
    sbert_df = prep(read_first_parquet(args.sbert), "sbert")

    # Drive merge by length_df rows
    total = len(length_df)
    print(f"Total rows (length): {total} | chunk size: {CHUNK_SIZE}")

    out_file = os.path.join(args.out, "data.parquet")

    # Streaming parquet writer (preferred)
    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        writer = None
        part = 0

        for start in range(0, total, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, total)
            base = length_df.iloc[start:end]
            idx = base.index

            # IMPORTANT: use .loc instead of reindex (because now indices are unique)
            merged = (
                base
                .join(sentiment_df.loc[idx], how="inner")
                .join(sbert_df.loc[idx], how="inner")
                .join(tfidf_df.loc[idx], how="inner")
                .reset_index()
            )

            merged["timestamp"] = pd.Timestamp("today").normalize()

            table = pa.Table.from_pandas(merged, preserve_index=False)

            if writer is None:
                writer = pq.ParquetWriter(out_file, table.schema)

            writer.write_table(table)
            print(f"Wrote chunk {part} | rows {start}..{end} | shape={merged.shape}")

            del base, merged, table
            part += 1

        if writer is not None:
            writer.close()

        print(f"Final output written: {out_file}")

    except Exception as e:
        print(f"pyarrow streaming not available ({e}). Falling back to part files...")
        part_dir = os.path.join(args.out, "parts")
        os.makedirs(part_dir, exist_ok=True)

        part = 0
        for start in range(0, total, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, total)
            base = length_df.iloc[start:end]
            idx = base.index

            merged = (
                base
                .join(sentiment_df.loc[idx], how="inner")
                .join(sbert_df.loc[idx], how="inner")
                .join(tfidf_df.loc[idx], how="inner")
                .reset_index()
            )

            merged["timestamp"] = pd.Timestamp("today").normalize()

            merged.to_parquet(os.path.join(part_dir, f"part-{part:05d}.parquet"), index=False)
            print(f"Wrote part {part} | rows {start}..{end} | shape={merged.shape}")

            del base, merged
            part += 1

        print(f"Output written as parts: {part_dir}")


if __name__ == "__main__":
    main()