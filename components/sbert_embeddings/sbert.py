import argparse
import os
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import IncrementalPCA

KEYS = ["asin", "reviewerID"]

def read_df(folder_path: str) -> pd.DataFrame:
    files = [f for f in os.listdir(folder_path) if f.endswith(".parquet")]
    if not files:
        raise FileNotFoundError(f"No parquet file found in {folder_path}")
    return pd.read_parquet(os.path.join(folder_path, files[0]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="reviewText")

    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--chunk_rows", type=int, default=25_000)  # chunk rows from dataframe
    parser.add_argument("--pca_components", type=int, default=64)
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    tmp_dir = os.path.join(args.out, "_tmp_sbert")
    os.makedirs(tmp_dir, exist_ok=True)

    df = read_df(args.data)

    # Keep only what we need to reduce memory
    df = df[KEYS + [args.text_column]].copy()
    df[args.text_column] = df[args.text_column].fillna("").astype(str)

    n = len(df)
    print(f"Rows: {n}")

    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

    # -------------------------
    # PASS 1: encode to disk
    # -------------------------
    emb_files = []
    key_files = []
    part = 0

    for start in range(0, n, args.chunk_rows):
        end = min(start + args.chunk_rows, n)
        chunk = df.iloc[start:end]

        texts = chunk[args.text_column].tolist()
        keys = chunk[KEYS].reset_index(drop=True)

        emb = model.encode(
            texts,
            batch_size=args.batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=False,
        ).astype("float32")

        emb_path = os.path.join(tmp_dir, f"emb-{part:05d}.npy")
        keys_path = os.path.join(tmp_dir, f"keys-{part:05d}.parquet")

        np.save(emb_path, emb)
        keys.to_parquet(keys_path, index=False)

        emb_files.append(emb_path)
        key_files.append(keys_path)

        print(f"Encoded part {part} | rows {start}..{end} | emb shape={emb.shape}")
        del chunk, texts, keys, emb
        part += 1

    # -------------------------
    # PASS 2: fit IncrementalPCA
    # -------------------------
    ipca = IncrementalPCA(n_components=args.pca_components)

    for i, emb_path in enumerate(emb_files):
        emb = np.load(emb_path, mmap_mode="r")  # memory-mapped
        ipca.partial_fit(emb)
        print(f"IPCA partial_fit on part {i} | shape={emb.shape}")

    # -------------------------
    # PASS 3: transform + write output (streaming parquet)
    # -------------------------
    out_file = os.path.join(args.out, "data.parquet")

    try:
        import pyarrow as pa
        import pyarrow.parquet as pq

        writer = None
        for i, (emb_path, keys_path) in enumerate(zip(emb_files, key_files)):
            emb = np.load(emb_path, mmap_mode="r")
            reduced = ipca.transform(emb).astype("float32")

            keys = pd.read_parquet(keys_path)
            feat = pd.DataFrame(reduced, columns=[f"sbert_{j}" for j in range(reduced.shape[1])])
            out_df = pd.concat([keys, feat], axis=1)

            table = pa.Table.from_pandas(out_df, preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(out_file, table.schema)
            writer.write_table(table)

            print(f"Wrote reduced part {i} | shape={out_df.shape}")
            del emb, reduced, keys, feat, out_df, table

        if writer is not None:
            writer.close()

        print(f"SBERT output written (streaming): {out_file}")

    except Exception as e:
        # fallback: parts folder
        print(f"pyarrow streaming not available ({e}). Falling back to parts...")
        part_dir = os.path.join(args.out, "parts")
        os.makedirs(part_dir, exist_ok=True)

        for i, (emb_path, keys_path) in enumerate(zip(emb_files, key_files)):
            emb = np.load(emb_path, mmap_mode="r")
            reduced = ipca.transform(emb).astype("float32")

            keys = pd.read_parquet(keys_path)
            feat = pd.DataFrame(reduced, columns=[f"sbert_{j}" for j in range(reduced.shape[1])])
            out_df = pd.concat([keys, feat], axis=1)

            out_df.to_parquet(os.path.join(part_dir, f"part-{i:05d}.parquet"), index=False)
            print(f"Wrote reduced part {i} | shape={out_df.shape}")
            del emb, reduced, keys, feat, out_df

        print(f"SBERT output written as parts: {part_dir}")

    # Optional cleanup (comment out if you want to keep temp files for debugging)
    # import shutil
    # shutil.rmtree(tmp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()