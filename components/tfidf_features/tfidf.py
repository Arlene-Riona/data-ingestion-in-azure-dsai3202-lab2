import argparse
import os
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

KEYS = ["asin", "reviewerID"]

def read_df(folder_path: str) -> pd.DataFrame:
    return pd.read_parquet(os.path.join(folder_path, "data.parquet"))

def write_df(df: pd.DataFrame, out_folder: str):
    os.makedirs(out_folder, exist_ok=True)
    df.to_parquet(os.path.join(out_folder, "data.parquet"), index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", type=str, required=True)
    parser.add_argument("--val", type=str, required=True)
    parser.add_argument("--test", type=str, required=True)
    parser.add_argument("--text_column", type=str, default="reviewText")

    # Recommended settings
    parser.add_argument("--max_features", type=int, default=500)
    parser.add_argument("--ngram_min", type=int, default=1)
    parser.add_argument("--ngram_max", type=int, default=2)

    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    args = parser.parse_args()

    train_df = read_df(args.train)
    val_df = read_df(args.val)
    test_df = read_df(args.test)

    train_text = train_df[args.text_column].fillna("").astype(str)
    val_text = val_df[args.text_column].fillna("").astype(str)
    test_text = test_df[args.text_column].fillna("").astype(str)

    # IMPORTANT: fit ONLY on training split
    vec = TfidfVectorizer(
        max_features=args.max_features,
        stop_words="english",
        ngram_range=(args.ngram_min, args.ngram_max),
    )

    X_train = vec.fit_transform(train_text)
    X_val = vec.transform(val_text)
    X_test = vec.transform(test_text)

    feat_cols = [f"tfidf_{i}" for i in range(X_train.shape[1])]

    out_train = pd.DataFrame(X_train.toarray(), columns=feat_cols).astype("float32")
    out_val = pd.DataFrame(X_val.toarray(), columns=feat_cols).astype("float32")
    out_test = pd.DataFrame(X_test.toarray(), columns=feat_cols).astype("float32")

    # Keep keys for downstream merge
    out_train[KEYS] = train_df[KEYS].reset_index(drop=True)
    out_val[KEYS] = val_df[KEYS].reset_index(drop=True)
    out_test[KEYS] = test_df[KEYS].reset_index(drop=True)

    write_df(out_train, args.train_out)
    write_df(out_val, args.val_out)
    write_df(out_test, args.test_out)

    print("TF-IDF features:", len(feat_cols))

if __name__ == "__main__":
    main()