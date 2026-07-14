"""
IELTS Academic EN-VI data processing pipeline.

Origin: prepare_acacdemic_data.ipynb (Colab)
Converted to: local script for RTX 4050

Processing flow:
  1. Read .txt EN/VI files from the academic_sentence directory
  2. Pair files by prefix (001_en.txt <-> 001_vi.txt)
  3. Split into sentences, classify mismatch level
  4. Alignment:
     - diff=0  -> direct 1-1 pairing
     - diff 1-2 -> Gale-Church alignment
     - diff>=3  -> LaBSE semantic alignment (GPU)
  5. Filter bad pairs (semantic_score < threshold), dedup
  6. Split train/valid/test -> save as Parquet + ZIP

Usage:
  python -m datapipeline.prepare_academic_data --input-dir "D:/Studying/academic_sentence"
  python -m datapipeline.prepare_academic_data --input-dir "D:/Studying/academic_sentence" --output-dir ./data/academic --threshold 0.40
"""

import os
import sys
import re
import argparse
import shutil
import zipfile
import numpy as np
import pandas as pd
from collections import Counter

from datapipeline.utils import read_file, load_labse

# Fix encoding for Windows console
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")


# Sentence splitting

def split_sentences(text: str) -> list[str]:
    """
    Split text into sentences by terminal punctuation,
    while protecting abbreviations (Mr., Mrs., e.g., i.e., etc.).
    """
    text = re.sub(
        r'\b(Mr|Mrs|Ms|Dr|Prof|e\.g|i\.e|etc|vs|et al|Fig|Ref|Vol|Eq|Ph\.D|Inc|Ltd|Co|Corp|St)\.',
        lambda m: m.group(0).replace('.', '<DOT>'), text
    )
    sents = re.split(r'(?<=[.!?])\s+(?=[A-ZÀ-Ỹ"\'\u201c\u2018(])', text)
    sents = [s.replace('<DOT>', '.').strip() for s in sents if s.strip()]
    return sents


# Gale-Church Alignment (for diff 1-2 sentences)

def gale_church_align(en_sents: list[str], vi_sents: list[str]) -> list[tuple]:
    """
    Length-based sentence alignment (Gale-Church Algorithm / Length-based DP).
    Suitable for texts with small sentence count mismatch (1-2 sentences).
    """
    n, m = len(en_sents), len(vi_sents)
    if n == 0 or m == 0:
        return []

    len_en = [len(s) for s in en_sents]
    len_vi = [len(s) for s in vi_sents]

    # Average character ratio between VI and EN
    c = sum(len_vi) / max(sum(len_en), 1)

    def match_cost(l_en, l_vi):
        expected = l_en * c
        if expected == 0 and l_vi == 0:
            return 0
        ratio = min(expected, l_vi) / max(expected, l_vi)
        return (1.0 - ratio) * 10

    # DP matrix
    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    backtrack = {}

    # Alignment operations (di, dj, penalty)
    operations = [
        (1, 1, 0),      # 1-1 (highest priority)
        (2, 1, 2.5),    # 2 EN - 1 VI
        (1, 2, 2.5),    # 1 EN - 2 VI
        (2, 2, 3.5),    # 2-2
        (3, 1, 5.0),    # 3 EN - 1 VI
        (1, 3, 5.0),    # 1 EN - 3 VI
    ]

    for i in range(n + 1):
        for j in range(m + 1):
            if D[i, j] == np.inf:
                continue
            for di, dj, penalty in operations:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                l_en = sum(len_en[i:ni])
                l_vi = sum(len_vi[j:nj])
                cost = match_cost(l_en, l_vi) + penalty
                if D[i, j] + cost < D[ni, nj]:
                    D[ni, nj] = D[i, j] + cost
                    backtrack[(ni, nj)] = (i, j, di, dj)

    # Backtrack
    aligned_pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if (i, j) not in backtrack:
            break
        prev_i, prev_j, di, dj = backtrack[(i, j)]
        en_text = " ".join(en_sents[prev_i:i]).strip()
        vi_text = " ".join(vi_sents[prev_j:j]).strip()
        length_ratio = min(len(en_text), max(1, len(vi_text))) / max(len(en_text), max(1, len(vi_text)))
        aligned_pairs.append((en_text, vi_text, round(length_ratio, 3), f"{di}-{dj}", np.nan))
        i, j = prev_i, prev_j

    return aligned_pairs[::-1]


# LaBSE Semantic Alignment (for diff >= 3 sentences)

def semantic_align(en_sents: list[str], vi_sents: list[str],
                   # SentenceTransformer type (lazy import)
                   model) -> list[tuple]:
    """
    Semantic alignment using LaBSE for heavily misaligned segments (>= 3 sentences).
    Computes similarity matrix + shortest path DP.
    """
    n, m = len(en_sents), len(vi_sents)
    if n == 0 or m == 0:
        return []

    en_texts, vi_texts = [], []
    en_idx_map, vi_idx_map = {}, {}

    for i in range(n):
        for di in range(1, 4):
            if i + di <= n:
                en_idx_map[(i, di)] = len(en_texts)
                en_texts.append(" ".join(en_sents[i:i+di]).strip())

    for j in range(m):
        for dj in range(1, 4):
            if j + dj <= m:
                vi_idx_map[(j, dj)] = len(vi_texts)
                vi_texts.append(" ".join(vi_sents[j:j+dj]).strip())

    # Encode on GPU
    from sentence_transformers import util
    en_embs = model.encode(en_texts, convert_to_tensor=True, show_progress_bar=False)
    vi_embs = model.encode(vi_texts, convert_to_tensor=True, show_progress_bar=False)
    sim_matrix = util.cos_sim(en_embs, vi_embs)

    D = np.full((n + 1, m + 1), np.inf)
    D[0, 0] = 0
    backtrack = {}

    operations = [
        (1, 1, 0.0),
        (2, 1, 0.1), (1, 2, 0.1),
        (3, 1, 0.2), (1, 3, 0.2),
        (2, 2, 0.3),
        (1, 0, 0.7), (0, 1, 0.7),
    ]

    for i in range(n + 1):
        for j in range(m + 1):
            if D[i, j] == np.inf:
                continue
            for di, dj, penalty in operations:
                ni, nj = i + di, j + dj
                if ni > n or nj > m:
                    continue
                sim_val = 0.0
                if di > 0 and dj > 0:
                    idx_en = en_idx_map[(i, di)]
                    idx_vi = vi_idx_map[(j, dj)]
                    sim_val = sim_matrix[idx_en][idx_vi].item()
                    step_cost = (1.0 - sim_val) + penalty
                    # Penalize low similarity
                    if sim_val < 0.30:
                        step_cost += 2.0
                    if sim_val < 0.20:
                        step_cost += 5.0
                else:
                    step_cost = penalty

                if D[i, j] + step_cost < D[ni, nj]:
                    D[ni, nj] = D[i, j] + step_cost
                    backtrack[(ni, nj)] = (i, j, di, dj, sim_val)

    aligned_pairs = []
    i, j = n, m
    while i > 0 or j > 0:
        if (i, j) not in backtrack:
            break
        prev_i, prev_j, di, dj, sim_val = backtrack[(i, j)]
        en_text = " ".join(en_sents[prev_i:i]).strip()
        vi_text = " ".join(vi_sents[prev_j:j]).strip()
        if en_text and vi_text:
            length_ratio = min(len(en_text), len(vi_text)) / max(len(en_text), len(vi_text))
            aligned_pairs.append((en_text, vi_text, round(length_ratio, 3),
                                  f"{di}-{dj}", round(sim_val, 4)))
        i, j = prev_i, prev_j

    return aligned_pairs[::-1]


# Main pipeline

def collect_txt_files(input_dir: str) -> list[str]:
    """Collect all .txt files from a directory."""
    all_txt = []
    for root, dirs, files in os.walk(input_dir):
        for f in files:
            if f.endswith(".txt"):
                all_txt.append(os.path.join(root, f))
    return all_txt


def pair_files(all_txt: list[str]) -> tuple[list, dict, dict]:
    """Classify EN/VI files and pair them by prefix."""
    en_files = sorted([f for f in all_txt if "_en.txt" in os.path.basename(f)])
    vi_files = sorted([f for f in all_txt if "_vi.txt" in os.path.basename(f)])

    def get_prefix(path):
        return os.path.basename(path).split("_")[0]

    en_dict = {get_prefix(f): f for f in en_files}
    vi_dict = {get_prefix(f): f for f in vi_files}
    common_keys = sorted(set(en_dict.keys()) & set(vi_dict.keys()))

    print(f"  Total .txt files: {len(all_txt)}")
    print(f"  EN files: {len(en_files)} | VI files: {len(vi_files)}")
    print(f"  Paired: {len(common_keys)}")

    return common_keys, en_dict, vi_dict


def load_and_pair(input_dir: str) -> list[dict]:
    """Read and pair EN/VI content into a list of dicts."""
    all_txt = collect_txt_files(input_dir)
    common_keys, en_dict, vi_dict = pair_files(all_txt)

    data = []
    for key in common_keys:
        en_text = read_file(en_dict[key]).strip()
        vi_text = read_file(vi_dict[key]).strip()
        data.append({"id": key, "en": en_text, "vi": vi_text})

    print(f"  Pairs loaded: {len(data)}")
    return data


def align_all(data: list[dict], model, threshold: float) -> list[dict]:
    """
    Split sentences, classify by mismatch level, align, and filter by threshold.

    Args:
        data: list[dict] with keys "id", "en", "vi"
        model: LaBSE SentenceTransformer model
        threshold: minimum semantic_score threshold (applied to LaBSE pairs)
    """
    # Step 1: Split sentences and classify
    alignment_data = []
    for d in data:
        en_sents = split_sentences(d["en"])
        vi_sents = split_sentences(d["vi"])
        diff = len(en_sents) - len(vi_sents)
        alignment_data.append({
            "id": d["id"],
            "en_count": len(en_sents),
            "vi_count": len(vi_sents),
            "diff": diff,
            "abs_diff": abs(diff),
            "en_sents": en_sents,
            "vi_sents": vi_sents,
        })

    # Statistics
    diffs = [a["abs_diff"] for a in alignment_data]
    diff_counter = Counter(diffs)
    perfect = diff_counter[0]
    gc = sum(diff_counter[k] for k in diff_counter if 1 <= k <= 2)
    labse = sum(diff_counter[k] for k in diff_counter if k >= 3)
    print(f"\n  Sentence mismatch distribution:")
    print(f"    Match (diff=0):    {perfect} pairs")
    print(f"    Diff 1-2 sents:    {gc} pairs  -> Gale-Church")
    print(f"    Diff >=3 sents:    {labse} pairs  -> LaBSE")

    # Step 2: Alignment
    final_data = []
    for idx, a in enumerate(alignment_data):
        file_id = a["id"]
        en_sents = a["en_sents"]
        vi_sents = a["vi_sents"]
        abs_diff = a["abs_diff"]

        # Exact match -> direct 1-1 pairing
        if abs_diff == 0:
            for i in range(len(en_sents)):
                en_text = en_sents[i].strip()
                vi_text = vi_sents[i].strip()
                if not en_text or not vi_text:
                    continue
                length_ratio = min(len(en_text), len(vi_text)) / max(len(en_text), len(vi_text))
                final_data.append({
                    "file_id": file_id,
                    "en": en_text,
                    "vi": vi_text,
                    "length_ratio": round(length_ratio, 3),
                    "alignment_type": "1-1",
                    "semantic_score": 1.0,
                    "method": "Direct",
                })

        # Small mismatch -> Gale-Church
        elif 1 <= abs_diff <= 2:
            aligned_pairs = gale_church_align(en_sents, vi_sents)
            for pair in aligned_pairs:
                en_text, vi_text, length_ratio, align_type, _ = pair
                final_data.append({
                    "file_id": file_id,
                    "en": en_text,
                    "vi": vi_text,
                    "length_ratio": length_ratio,
                    "alignment_type": align_type,
                    "semantic_score": np.nan,
                    "method": "Gale-Church",
                })

        # Heavy mismatch -> LaBSE semantic alignment
        else:
            aligned_pairs = semantic_align(en_sents, vi_sents, model)
            for pair in aligned_pairs:
                en_text, vi_text, length_ratio, align_type, sim_score = pair
                final_data.append({
                    "file_id": file_id,
                    "en": en_text,
                    "vi": vi_text,
                    "length_ratio": length_ratio,
                    "alignment_type": align_type,
                    "semantic_score": sim_score,
                    "method": "LaBSE",
                })

    print(f"  Total raw pairs: {len(final_data)}")

    # Step 3: Clean and filter
    df = pd.DataFrame(final_data)
    df.replace("", np.nan, inplace=True)
    df.dropna(subset=["en", "vi"], inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Filter bad pairs by threshold (only applies to LaBSE group)
    before = len(df)
    df = df[
        ~((df["method"] == "LaBSE") & (df["semantic_score"] < threshold))
    ]
    print(f"  After threshold {threshold} filter (LaBSE): {before} -> {len(df)}")

    # Dedup
    before = len(df)
    df.drop_duplicates(subset=["en", "vi"], inplace=True)
    df.reset_index(drop=True, inplace=True)
    print(f"  After dedup: {before} -> {len(df)}")

    return df


def split_and_save(df: pd.DataFrame, output_dir: str, seed: int = 42):
    from sklearn.model_selection import train_test_split
    """Split train/valid/test and save as Parquet + ZIP."""
    os.makedirs(output_dir, exist_ok=True)

    # Split: reserve 1000 samples for valid+test, rest is train
    n_holdout = min(1000, len(df) // 5)  # max 1000 or 20% of data
    df_train, df_temp = train_test_split(
        df, test_size=n_holdout / len(df), random_state=seed
    )
    df_valid, df_test = train_test_split(df_temp, test_size=0.5, random_state=seed)

    print(f"\n  Split results:")
    print(f"    Train:      {len(df_train):,} ({len(df_train)/len(df):.1%})")
    print(f"    Validation: {len(df_valid):,} ({len(df_valid)/len(df):.1%})")
    print(f"    Test:       {len(df_test):,} ({len(df_test)/len(df):.1%})")

    # Save Parquet
    paths = {}
    for name, split_df in [("train", df_train), ("valid", df_valid), ("test", df_test)]:
        p = os.path.join(output_dir, f"ielts_academic_{name}.parquet")
        split_df.to_parquet(p, engine="pyarrow", index=False)
        paths[name] = p

    # Save full clean dataset
    clean_path = os.path.join(output_dir, "ielts_academic_clean.parquet")
    df.to_parquet(clean_path, engine="pyarrow", index=False)
    paths["clean"] = clean_path

    # Compress to ZIP
    zip_path = os.path.join(output_dir, "ielts_academic_splits.zip")
    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zipf:
        for p in paths.values():
            zipf.write(p, os.path.basename(p))
    print(f"  Saved ZIP: {zip_path}")

    return paths


# Entry point

def parse_args():
    parser = argparse.ArgumentParser(
        description="IELTS Academic EN-VI data processing pipeline"
    )
    parser.add_argument(
        "--input-dir", type=str,
        default=r"D:\Studying\academic_sentence",
        help="Directory containing EN/VI .txt files (default: D:\\Studying\\academic_sentence)"
    )
    parser.add_argument(
        "--output-dir", type=str,
        default="./data/academic_output",
        help="Output directory (default: ./data/academic_output)"
    )
    parser.add_argument(
        "--threshold", type=float, default=0.40,
        help="Minimum semantic_score threshold for LaBSE pairs (default: 0.40)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print("=" * 60)
    print("  PIPELINE: Prepare IELTS Academic Data")
    print("=" * 60)
    print(f"  Input:     {args.input_dir}")
    print(f"  Output:    {args.output_dir}")
    print(f"  Threshold: {args.threshold}")
    print(f"  Seed:      {args.seed}")
    print()

    # 1. Load LaBSE
    print("[1/4] Loading LaBSE model...")
    model = load_labse()

    # 2. Read and pair files
    print("\n[2/4] Reading and pairing EN/VI files...")
    data = load_and_pair(args.input_dir)

    # 3. Align and filter
    print("\n[3/4] Aligning sentences and filtering...")
    df = align_all(data, model, threshold=args.threshold)

    # 4. Split and save
    print("\n[4/4] Splitting and saving results...")
    paths = split_and_save(df, args.output_dir, seed=args.seed)

    print("\n" + "=" * 60)
    print("  DONE!")
    print(f"  Total pairs: {len(df):,}")
    print(f"  Output: {args.output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
