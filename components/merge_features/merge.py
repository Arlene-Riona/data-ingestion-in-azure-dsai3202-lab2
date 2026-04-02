'''
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
    before = len(df)
    dup_count = df.duplicated(subset=KEYS).sum()
    if dup_count > 0:
        print(f"{name}: found {dup_count} duplicate keys. Deduplicating (keep='last')...")
        df = df.drop_duplicates(subset=KEYS, keep="last")
    after = len(df)
    print(f"{name}: rows {before} -> {after}")
    return df


def prep(df: pd.DataFrame, name: str, keep_cols=None) -> pd.DataFrame:
    df = downcast(df)
    df = make_unique(df, name)
    # If there are extra columns to keep alongside the index, store them separately
    if keep_cols:
        extras = df[KEYS + keep_cols].copy()
        df = df.set_index(KEYS).sort_index()
        return df, extras
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

    # Load length df and preserve 'overall' separately
    length_raw = read_first_parquet(args.length)
    length_df = prep(length_raw, "length")

    # pull label directly from raw input safely
    overall_df = length_raw[["asin", "reviewerID", "overall"]].copy()

    sentiment_df = prep(read_first_parquet(args.sentiment), "sentiment")
    tfidf_df     = prep(read_first_parquet(args.tfidf),     "tfidf")
    sbert_df     = prep(read_first_parquet(args.sbert),     "sbert")

    total = len(length_df)
    print(f"Total rows (length): {total} | chunk size: {CHUNK_SIZE}")

    out_file = os.path.join(args.out, "data.parquet")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        writer = None
        part = 0

        for start in range(0, total, CHUNK_SIZE):
            end = min(start + CHUNK_SIZE, total)
            base = length_df.iloc[start:end]
            idx = base.index

            merged = (
                base
                .join(sentiment_df.loc[idx], how="inner")
                .join(sbert_df.loc[idx],     how="inner")
                .join(tfidf_df.loc[idx],     how="inner")
                .reset_index()
            )

            # Add overall back by merging on KEYS
            merged = merged.merge(overall_df, on=KEYS, how="left")

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
                .join(sbert_df.loc[idx],     how="inner")
                .join(tfidf_df.loc[idx],     how="inner")
                .reset_index()
            )

            merged = merged.merge(overall_df, on=KEYS, how="left")
            merged["timestamp"] = pd.Timestamp("today").normalize()

            merged.to_parquet(
                os.path.join(part_dir, f"part-{part:05d}.parquet"), index=False
            )
            print(f"Wrote part {part} | rows {start}..{end} | shape={merged.shape}")

            del base, merged
            part += 1

        print(f"Output written as parts: {part_dir}")


if __name__ == "__main__":
    main()
'''
import argparse
import os
import pandas as pd
import gc

KEYS = ["asin", "reviewerID"]
CHUNK_SIZE = 5000 


def read_parquet(folder_path):
    files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet file in {folder_path}")
    return pd.read_parquet(os.path.join(folder_path, files[0]))


def main(length_path, sentiment_path, tfidf_path, sbert_path, raw_path, output_path):

    print("Loading datasets...")

    length_df = read_parquet(length_path)
    sentiment_df = read_parquet(sentiment_path)
    tfidf_df = read_parquet(tfidf_path)
    sbert_df = read_parquet(sbert_path)
    raw_df = read_parquet(raw_path)

    # -----------------------
    # Prepare labels
    # -----------------------
    labels_df = raw_df[KEYS + ["overall"]].copy()

    # Set index for fast joins
    length_df = length_df.set_index(KEYS)
    sentiment_df = sentiment_df.set_index(KEYS)
    tfidf_df = tfidf_df.set_index(KEYS)
    sbert_df = sbert_df.set_index(KEYS)
    labels_df = labels_df.set_index(KEYS)

    total = len(length_df)
    print(f"Total rows: {total}")

    os.makedirs(output_path, exist_ok=True)
    out_file = os.path.join(output_path, "data.parquet")

    parts = []
    part_id = 0

    # -----------------------
    # Chunked merge
    # -----------------------
    for start in range(0, total, CHUNK_SIZE):
        end = min(start + CHUNK_SIZE, total)

        base = length_df.iloc[start:end]

        merged = (
            base
            .join(sentiment_df, how="inner")
            .join(tfidf_df, how="inner")
            .join(sbert_df, how="inner")
            .join(labels_df, how="left")  # ✅ keeps overall safely
            .reset_index()
        )

        if merged.empty:
            print(f"Chunk {part_id} is empty, skipping...")
            continue

        parts.append(merged)

        print(f"Processed chunk {part_id} | rows {start}-{end} | shape={merged.shape}")

        del base, merged
        gc.collect()

        part_id += 1

    # -----------------------
    # Final save
    # -----------------------
    final_df = pd.concat(parts, ignore_index=True)

    if "overall" not in final_df.columns:
        raise ValueError("overall column missing after merge")

    final_df.to_parquet(out_file, index=False)

    print("✅ Merge successful")
    print(f"Final shape: {final_df.shape}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--length", required=True)
    parser.add_argument("--sentiment", required=True)
    parser.add_argument("--tfidf", required=True)
    parser.add_argument("--sbert", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--output", required=True)

    args = parser.parse_args()

    main(
        args.length,
        args.sentiment,
        args.tfidf,
        args.sbert,
        args.raw,
        args.output
    )