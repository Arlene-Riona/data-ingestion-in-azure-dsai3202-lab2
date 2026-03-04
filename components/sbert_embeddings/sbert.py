import argparse
import os
import pandas as pd
from sentence_transformers import SentenceTransformer

KEYS = ["asin", "reviewerID"]

def read_df(folder_path: str) -> pd.DataFrame:
    # Option B: Look for any .parquet file in the folder
    files = [f for f in os.listdir(folder_path) if f.endswith('.parquet')]
    if not files:
        raise FileNotFoundError(f"No parquet file found in {folder_path}")
    return pd.read_parquet(os.path.join(folder_path, files[0]))

def write_df(df: pd.DataFrame, out_folder: str):
    os.makedirs(out_folder, exist_ok=True)
    # Keeping 'data.parquet' for the next step (Merge)
    df.to_parquet(os.path.join(out_folder, "data.parquet"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--out", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="reviewText")
    parser.add_argument("--batch_size", type=int, default=64)
    args = parser.parse_args()

    df = read_df(args.data)
    texts = df[args.text_column].fillna("").astype(str).tolist()

    # Load model (Note: This will download the model to the cluster)
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    emb = model.encode(texts, batch_size=args.batch_size, show_progress_bar=False)

    out_df = df[KEYS].copy()
    out_df["bert_embedding"] = [e.tolist() for e in emb]

    write_df(out_df, args.out)
    
    if len(out_df) > 0:
        print("Embedding dim:", len(out_df["bert_embedding"].iloc[0]))
    else:
        print("Empty dataframe processed.")

if __name__ == "__main__":
    main()