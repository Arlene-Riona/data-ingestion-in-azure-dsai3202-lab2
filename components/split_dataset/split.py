import argparse
import os
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--train_out", type=str, required=True)
    parser.add_argument("--val_out", type=str, required=True)
    parser.add_argument("--test_out", type=str, required=True)
    parser.add_argument("--deploy_out", type=str, required=True)
    return parser.parse_args()


def main():
    args = parse_args()

    files = [f for f in os.listdir(args.data) if f.endswith('.parquet')]
    if not files:
        raise FileNotFoundError(f"No parquet file found in {args.data}")

    input_path = os.path.join(args.data, files[0])
    print(f"Reading dataset from: {input_path}")
    df = pd.read_parquet(input_path)

    # Sort by review_year so deploy comes from most recent data
    df = df.sort_values('review_year').reset_index(drop=True)

    # Take most recent 10% as deploy split
    split_idx = int(len(df) * 0.90)
    df_main = df.iloc[:split_idx]
    df_deploy = df.iloc[split_idx:]

    # Split remaining 90% into train/val/test (60/15/15 of total)
    df_train, df_temp = train_test_split(
        df_main,
        test_size=0.333,
        random_state=args.seed,
        shuffle=True
    )
    df_val, df_test = train_test_split(
        df_temp,
        test_size=0.5,
        random_state=args.seed,
        shuffle=True
    )

    # Write outputs
    for path, data, name in [
        (args.train_out, df_train, "Train"),
        (args.val_out, df_val, "Val"),
        (args.test_out, df_test, "Test"),
        (args.deploy_out, df_deploy, "Deploy"),
    ]:
        os.makedirs(path, exist_ok=True)
        data.to_parquet(os.path.join(path, "data.parquet"), index=False)
        print(f"{name} rows: {len(data)}")


if __name__ == "__main__":
    main()