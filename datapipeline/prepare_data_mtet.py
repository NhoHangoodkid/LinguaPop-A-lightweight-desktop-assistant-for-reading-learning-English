"""
MTET (Machine Translation EN-VI) data processing pipeline.

Origin: prepare_data_mtet.ipynb (Colab)
Converted to: local script for RTX 4050

Processing flow:
  1. Load dataset hiimbach/mtet from Hugging Face
  2. Clean + filter bad pairs + dedup
  3. Compute LaBSE cross-lingual similarity (GPU, batch=128 for RTX 4050)
  4. Apply single similarity threshold
  5. Split valid/test from proc_test, remove overlap with train
  6. Save train/valid/test/pool as Parquet + ZIP

Usage:
  python -m datapipeline.prepare_data_mtet
  python -m datapipeline.prepare_data_mtet --threshold 0.58 --output-dir ./data/mtet_output
  python -m datapipeline.prepare_data_mtet --n-train 230000 --n-valid 15000 --n-test 25000
"""

import os
import sys
import argparse
import shutil
import numpy as np
import pandas as pd

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from datasets import load_dataset

from datapipeline.utils import (
    clean_series,
    is_bad_pair,
    filter_quality,
    dedup,
    apply_lang_filter,
    load_labse,
    add_cross_sim,
    remove_overlap,
)


# Process a single split

def process_split(df: pd.DataFrame, labse_model, name: str = "split",
                  sample_n: int = None, threshold: float = 0.58,
                  batch_size: int = 128) -> pd.DataFrame:
    """
    Full pipeline for a single split:
    clean -> filter bad pairs -> quality filter -> dedup -> (sample) -> LaBSE gate.
    """
    df = df.copy()
    df["vi"] = clean_series(df["vi"])
    df["en"] = clean_series(df["en"])

    # Quick filter
    before = len(df)
    keep = [not is_bad_pair(en, vi) for en, vi in zip(df["en"], df["vi"])]
    df = df[keep].reset_index(drop=True)
    df = filter_quality(df)
    df = dedup(df)
    df = apply_lang_filter(df)
    print(f"  [{name}] Quick filter: {before:,} -> {len(df):,}")

    # Sample if needed (reduce load before LaBSE encoding)
    if sample_n and len(df) > sample_n:
        df = df.sample(n=sample_n, random_state=42).reset_index(drop=True)
        print(f"  [{name}] Sampled down to: {len(df):,} before encoding")

    # Compute LaBSE similarity
    df = add_cross_sim(df, labse_model, batch_size=batch_size)

    # Log similarity statistics
    print(f"  [{name}] Similarity: median={df['sim'].median():.3f}, "
          f"mean={df['sim'].mean():.3f}, min={df['sim'].min():.3f}")

    # Apply threshold
    before = len(df)
    df = df[df["sim"] >= threshold].reset_index(drop=True)
    print(f"  [{name}] LaBSE gate @ {threshold}: {before:,} -> {len(df):,}\n")

    return df[["vi", "en", "sim"]]


# Main pipeline

def save_outputs(train_df, valid_df, test_df, pool_df, output_dir: str):
    """Save all splits as Parquet + compress to ZIP."""
    os.makedirs(output_dir, exist_ok=True)

    train_df.to_parquet(f"{output_dir}/train.parquet", index=False)
    valid_df.to_parquet(f"{output_dir}/valid.parquet", index=False)
    test_df.to_parquet(f"{output_dir}/test.parquet",   index=False)
    pool_df.to_parquet(f"{output_dir}/pool.parquet",   index=False)

    zip_filename = f"{output_dir}.zip"
    shutil.make_archive(output_dir, "zip", output_dir)
    print(f"  Saved ZIP: {zip_filename}")

    return zip_filename


# Entry point

def parse_args():
    parser = argparse.ArgumentParser(
        description="MTET EN-VI data processing pipeline"
    )
    parser.add_argument(
        "--output-dir", type=str, default="./data/mtet_output",
        help="Output directory (default: ./data/mtet_output)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.58,
        help="Single LaBSE similarity threshold for both train and test (default: 0.58)"
    )
    parser.add_argument(
        "--n-train", type=int, default=230000,
        help="Number of train samples (default: 230000)"
    )
    parser.add_argument(
        "--n-valid", type=int, default=15000,
        help="Number of validation samples (default: 15000)"
    )
    parser.add_argument(
        "--n-test", type=int, default=25000,
        help="Number of test samples (default: 25000)"
    )
    parser.add_argument(
        "--reverse", type=int, default=1200000,
        help="Max raw train samples to use (default: 1200000)"
    )
    parser.add_argument(
        "--labse-max-train", type=int, default=400000,
        help="Max samples before LaBSE encoding for train (default: 400000)"
    )
    parser.add_argument(
        "--labse-batch", type=int, default=128,
        help="Batch size for LaBSE encoding - use 128 for RTX 4050 (default: 128)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  PIPELINE: Prepare MTET Data")
    print("=" * 60)
    print(f"  Output:         {args.output_dir}")
    print(f"  Threshold:      {args.threshold}")
    print(f"  n_train:        {args.n_train:,}")
    print(f"  n_valid:        {args.n_valid:,}")
    print(f"  n_test:         {args.n_test:,}")
    print(f"  reverse:        {args.reverse:,}")
    print(f"  labse_max_train:{args.labse_max_train:,}")
    print(f"  labse_batch:    {args.labse_batch}")
    print(f"  seed:           {args.seed}")
    print()

    # 1. Load LaBSE
    print("[1/5] Loading LaBSE model...")
    labse = load_labse()

    # 2. Load raw data from Hugging Face
    print("\n[2/5] Loading dataset hiimbach/mtet...")
    dataset = load_dataset("hiimbach/mtet")
    raw_train = (
        pd.DataFrame(dataset["train"].shuffle(seed=args.seed))
        .iloc[:args.reverse]
        .reset_index(drop=True)
    )
    raw_test = (
        pd.DataFrame(dataset["test"].shuffle(seed=args.seed))
        .reset_index(drop=True)
    )
    print(f"  Raw train (trimmed to {args.reverse:,}): {len(raw_train):,}")
    print(f"  Raw test: {len(raw_test):,}\n")

    # 3. Process splits (using single threshold)
    print("[3/5] Processing splits...")
    proc_train = process_split(
        raw_train, labse, "train",
        sample_n=args.labse_max_train,
        threshold=args.threshold,
        batch_size=args.labse_batch,
    )
    proc_test = process_split(
        raw_test, labse, "test",
        threshold=args.threshold,
        batch_size=args.labse_batch,
    )

    # 4. Split valid/test and remove overlap
    print("[4/5] Splitting valid/test and removing overlap...")
    assert len(proc_test) >= args.n_valid + args.n_test, (
        f"Cleaned test only has {len(proc_test):,}, needs {args.n_valid + args.n_test:,}. "
        f"Try lowering threshold or n_valid/n_test."
    )
    valid_df = proc_test.iloc[:args.n_valid][["vi", "en"]].reset_index(drop=True)
    test_df = proc_test.iloc[args.n_valid:args.n_valid + args.n_test][["vi", "en"]].reset_index(drop=True)
    print(f"  Valid: {len(valid_df):,} | Test: {len(test_df):,}")

    # Remove train <-> valid/test overlap
    proc_train = remove_overlap(proc_train, valid_df, test_df)
    assert len(proc_train) >= args.n_train, (
        f"Cleaned train only has {len(proc_train):,} < n_train {args.n_train:,}. "
        f"Try increasing labse_max_train or lowering threshold."
    )

    proc_train = proc_train.sample(frac=1, random_state=args.seed).reset_index(drop=True)
    train_df = proc_train.iloc[:args.n_train][["vi", "en"]].reset_index(drop=True)
    pool_df = proc_train.iloc[args.n_train:][["vi", "en"]].reset_index(drop=True)
    print(f"  Train: {len(train_df):,} | Pool (reserve): {len(pool_df):,}\n")

    # 5. Save results
    print("[5/5] Saving results...")
    save_outputs(train_df, valid_df, test_df, pool_df, args.output_dir)

    print("\n" + "=" * 60)
    print("  DONE!")
    print(f"  Train: {len(train_df):,} | Valid: {len(valid_df):,} | "
          f"Test: {len(test_df):,} | Pool: {len(pool_df):,}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
