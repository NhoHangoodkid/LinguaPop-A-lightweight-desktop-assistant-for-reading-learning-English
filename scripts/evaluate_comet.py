"""
COMET Evaluation Script for MarianMT (EN -> VI) — CTranslate2 inference
========================================================================
Model format : CTranslate2 directory  (models/en-vi-ct2)
               converted using convert_to_ct2.py
Tokenizer    : MarianTokenizer directory (models/tokenizer-en-vi)
Test data    : Parquet file  (columns: "en", "vi")
COMET metric : wmt22-comet-da (reference-based)

Usage:
    python evaluate_comet.py \
        --ct2_dir     models/en-vi-ct2 \
        --tok_dir     models/tokenizer-en-vi \
        --test_file   data/test.parquet \
        --output_dir  comet_results \
        --save_translations

Optional flags:
    --src_col      en        (source column name in parquet, default: en)
    --tgt_col      vi        (reference column name in parquet, default: vi)
    --batch_size   16        (sentences per inference batch, default: 16)
    --beam_size    4         (beam search size, default: 4)
    --max_input    128       (truncate input tokens, default: 128)
    --max_target   128       (max generated tokens, default: 128)
    --inter_threads 2        (CT2 parallel threads, default: 2)
    --device       cuda      (cuda | cpu, auto-detected if omitted)
    --comet_batch  32        (batch size for COMET scorer, default: 32)
    --save_translations      (save translations to .txt file)
"""

import argparse
import json
import time
import unicodedata
from pathlib import Path
import sys
import traceback

if sys.stdout.encoding != 'utf-8':
    sys.stdout.reconfigure(encoding='utf-8')
if sys.stderr.encoding != 'utf-8':
    sys.stderr.reconfigure(encoding='utf-8')

import pandas as pd
import ctranslate2
import sentencepiece as spm
import torch
from transformers import AutoTokenizer
from comet import download_model, load_from_checkpoint

# Add project root to sys.path to import src.engine
sys.path.append(str(Path(__file__).resolve().parent.parent))
from src.engine import TranslationEngine
# Argument parsing
def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CTranslate2 MarianMT model with COMET wmt22-comet-da"
    )

    # Model (CT2 format)
    parser.add_argument("--ct2_dir", type=str, required=True,
                        help="CT2 model directory (output of convert_to_ct2.py, e.g.: models/en-vi-ct2)")
    parser.add_argument("--tok_dir", type=str, required=True,
                        help="Tokenizer directory (e.g.: models/tokenizer-en-vi)")

    # Data
    parser.add_argument("--test_file", type=str, required=True,
                        help="test.parquet file")
    parser.add_argument("--src_col",   type=str, default="en",
                        help="Source column name [default: en]")
    parser.add_argument("--tgt_col",   type=str, default="vi",
                        help="Reference column name [default: vi]")

    # CT2 inference
    parser.add_argument("--batch_size",    type=int, default=16,
                        help="Sentences per batch [default: 16, safe for RTX 4050 6GB]")
    parser.add_argument("--beam_size",     type=int, default=4,
                        help="Beam search size [default: 4]")
    parser.add_argument("--max_input",     type=int, default=128,
                        help="Input token limit [default: 128]")
    parser.add_argument("--max_target",    type=int, default=128,
                        help="Max generated tokens [default: 128]")
    parser.add_argument("--inter_threads", type=int, default=2,
                        help="CT2Translator inter_threads (increase = faster on CPU) [default: 2]")
    parser.add_argument("--device",        type=str, default=None,
                        help="cuda | cpu (auto-detected if omitted)")

    # COMET
    parser.add_argument("--comet_batch", type=int, default=32,
                        help="COMET scorer batch size [default: 32, suitable for RTX 4050]")

    # Output
    parser.add_argument("--output_dir",        type=str, default="./comet_results")
    parser.add_argument("--save_translations",  action="store_true",
                        help="Save translations to hypotheses.txt")

    return parser.parse_args()
# Data loading
def load_parquet(path: str, src_col: str, tgt_col: str):
    """Read parquet, return (sources, references) as list[str]."""
    print(f"[Data] Reading: {path}")
    df = pd.read_parquet(path)

    missing = [c for c in [src_col, tgt_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Columns not found: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    before = len(df)
    df = df.dropna(subset=[src_col, tgt_col])
    after = len(df)
    if before != after:
        print(f"[Data] Dropped {before - after} null rows. Remaining: {after:,}")

    sources    = df[src_col].astype(str).tolist()
    references = df[tgt_col].astype(str).tolist()
    print(f"[Data] Loaded {len(sources):,} sentence pairs.")
    return sources, references
# CTranslate2 inference
def translate_dataset(engine: TranslationEngine, sources: list[str], batch_size: int) -> list[str]:
    """
    Inference with CTranslate2 via TranslationEngine:
    Calls engine.translate(src) exactly for each sentence to pass through _pack layer,
    ensuring 100% exact reproduction of the production flow.
    """
    total      = len(sources)
    hypotheses = []

    print(f"\n[CT2] Translating {total:,} sentences "
          f"(batch={batch_size}, beam={engine.num_beams}, "
          f"max_input={engine.max_input_length}, max_target={engine.max_decode_length}) ...")

    t0 = time.time()
    for i, src in enumerate(sources):
        # Call exact production entry point
        translated = engine.translate(src)
        # engine.translate returns a list with 1 element (joined by \n if split by _pack)
        hypotheses.append(translated[0] if translated else "")

        done = i + 1
        # Still using batch_size to log progress to avoid console spam
        if done % batch_size == 0 or done == total:
            elapsed = time.time() - t0
            speed   = done / elapsed
            eta     = (total - done) / speed if speed > 0 else 0
            print(f"  [{done:>6}/{total:,}]  {elapsed:6.1f}s  ETA {eta:5.1f}s  "
                  f"({speed:.1f} sent/s)")

    total_time = time.time() - t0
    print(f"[CT2] Completed {total:,} sentences in {total_time:.1f}s "
          f"({total/total_time:.1f} sent/s)")
    return hypotheses
# COMET scoring
def run_comet(sources: list[str], hypotheses: list[str], references: list[str], comet_batch: int, device: str) -> object:
    """Download wmt22-comet-da and score."""
    print("\n[COMET] Loading wmt22-comet-da ...")
    comet_path  = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_path)

    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]

    gpus = 1 if device == "cuda" else 0
    print(f"[COMET] Scoring {len(data):,} segments (batch={comet_batch}, gpus={gpus}) ...")
    t0     = time.time()
    output = comet_model.predict(data, batch_size=comet_batch, gpus=gpus)
    print(f"[COMET] Completed in {time.time() - t0:.1f}s")
    return output
# Main
def main():
    args = parse_args()

    # Determine device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Setup] Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    #  1. Load data 
    sources, references = load_parquet(args.test_file, args.src_col, args.tgt_col)

    #  2. Load CT2 Translation Engine 
    print(f"[CT2] Loading engine from: {args.ct2_dir} and {args.tok_dir} (device={device})")
    engine = TranslationEngine(
        model_dir=args.ct2_dir,
        tokenizer_dir=args.tok_dir,
        device=device,
        max_input_length=args.max_input,
        num_beams=args.beam_size,
        max_decode_length=args.max_target,
        inter_threads=args.inter_threads
    )

    #  3. Translate 
    hypotheses = translate_dataset(
        engine, sources,
        batch_size=args.batch_size,
    )

    # Use test file name as prefix for output
    file_stem = Path(args.test_file).stem

    if args.save_translations:
        hyp_path = out_dir / f"{file_stem}_hypotheses.txt"
        hyp_path.write_text("\n".join(hypotheses), encoding="utf-8")
        print(f"[Output] Translations saved to: {hyp_path}")

    #  4. COMET scoring 
    assert len(hypotheses) == len(sources), \
        f"Critical Error: Number of translations ({len(hypotheses)}) differs from number of source sentences ({len(sources)})."
    comet_output   = run_comet(sources, hypotheses, references, args.comet_batch, device)
    segment_scores = comet_output.scores
    system_score   = comet_output.system_score

    #  5. Save results 
    detail_df = pd.DataFrame({
        "source":      sources,
        "hypothesis":  hypotheses,
        "reference":   references,
        "comet_score": segment_scores,
    }).sort_values("comet_score", ascending=True)  # worst -> best for easier error analysis

    detail_path = out_dir / f"{file_stem}_segment_scores.csv"
    detail_df.to_csv(detail_path, index=False, encoding="utf-8-sig")
    print(f"[Output] Segment scores: {detail_path}")

    summary = {
        "ct2_dir":      args.ct2_dir,
        "tok_dir":      args.tok_dir,
        "test_file":    args.test_file,
        "num_segments": len(sources),
        "comet_metric": "wmt22-comet-da",
        "system_score": round(system_score, 4),
        "seg_mean":     round(sum(segment_scores) / len(segment_scores), 4),
        "seg_min":      round(min(segment_scores), 4),
        "seg_max":      round(max(segment_scores), 4),
        "beam_size":    args.beam_size,
        "max_input":    args.max_input,
        "max_target":   args.max_target,
        "device":       device,
    }
    summary_path = out_dir / f"{file_stem}_comet_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

    #  6. Print results 
    print("\n" + "=" * 55)
    print("  COMET EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Metric        : wmt22-comet-da (reference-based)")
    print(f"  Segments      : {len(sources):,}")
    print(f"  System score  : {system_score:.4f}   <- main number to report")
    print(f"  Seg mean      : {summary['seg_mean']:.4f}")
    print(f"  Seg min/max   : {summary['seg_min']:.4f} / {summary['seg_max']:.4f}")
    print("=" * 55)
    print(f"\n  Results saved to: {out_dir.resolve()}")
    print(f"    - {summary_path.name}  - overall score")
    print(f"    - {detail_path.name}  - detailed sentence scores (worst -> best)")
    if args.save_translations:
        print(f"    - {hyp_path.name}      - model translations")
    print()


if __name__ == "__main__":
    main()
