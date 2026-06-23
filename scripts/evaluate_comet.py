"""
COMET Evaluation Script for MarianMT (EN -> VI) — CTranslate2 inference
========================================================================
Model format : CTranslate2 directory  (models/en-vi-ct2)
               đã convert bằng convert_to_ct2.py
Tokenizer    : MarianTokenizer directory (models/tokenizer-en-vi)
Test data    : Parquet file  (columns: "en", "vi")
COMET metric : wmt22-comet-da (reference-based)

Cách dùng:
    python evaluate_comet.py \
        --ct2_dir     models/en-vi-ct2 \
        --tok_dir     models/tokenizer-en-vi \
        --test_file   data/test.parquet \
        --output_dir  comet_results \
        --save_translations

Flags tuỳ chọn:
    --src_col      en        (tên cột source trong parquet, mặc định: en)
    --tgt_col      vi        (tên cột reference trong parquet, mặc định: vi)
    --batch_size   16        (số câu mỗi batch inference, mặc định: 16)
    --beam_size    4         (beam search, mặc định: 4)
    --max_input    128       (cắt token đầu vào, mặc định: 128)
    --max_target   128       (số token tối đa sinh ra, mặc định: 128)
    --inter_threads 2        (số luồng song song CT2, mặc định: 2)
    --device       cuda      (cuda | cpu, tự detect nếu bỏ trống)
    --comet_batch  32        (batch size cho COMET scorer, mặc định: 32)
    --save_translations      (lưu bản dịch ra file .txt)
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


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate CTranslate2 MarianMT model với COMET wmt22-comet-da"
    )

    # Model (CT2 format)
    parser.add_argument("--ct2_dir", type=str, required=True,
                        help="Thư mục CT2 model (output của convert_to_ct2.py, vd: models/en-vi-ct2)")
    parser.add_argument("--tok_dir", type=str, required=True,
                        help="Thư mục tokenizer (vd: models/tokenizer-en-vi)")

    # Data
    parser.add_argument("--test_file", type=str, required=True,
                        help="File test.parquet")
    parser.add_argument("--src_col",   type=str, default="en",
                        help="Tên cột source  [mặc định: en]")
    parser.add_argument("--tgt_col",   type=str, default="vi",
                        help="Tên cột reference  [mặc định: vi]")

    # CT2 inference
    parser.add_argument("--batch_size",    type=int, default=16,
                        help="Số câu mỗi batch  [mặc định: 16, an toàn cho RTX 4050 6GB]")
    parser.add_argument("--beam_size",     type=int, default=4,
                        help="Beam search size  [mặc định: 4]")
    parser.add_argument("--max_input",     type=int, default=128,
                        help="Giới hạn token đầu vào  [mặc định: 128]")
    parser.add_argument("--max_target",    type=int, default=128,
                        help="Số token tối đa sinh ra  [mặc định: 128]")
    parser.add_argument("--inter_threads", type=int, default=2,
                        help="inter_threads của CT2Translator (tăng = nhanh hơn trên CPU)  [mặc định: 2]")
    parser.add_argument("--device",        type=str, default=None,
                        help="cuda | cpu  (tự detect nếu bỏ trống)")

    # COMET
    parser.add_argument("--comet_batch", type=int, default=32,
                        help="Batch size COMET scorer  [mặc định: 32, phù hợp RTX 4050]")

    # Output
    parser.add_argument("--output_dir",        type=str, default="./comet_results")
    parser.add_argument("--save_translations",  action="store_true",
                        help="Lưu bản dịch ra hypotheses.txt")

    return parser.parse_args()


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_parquet(path: str, src_col: str, tgt_col: str):
    """Đọc parquet, trả về (sources, references) dạng list[str]."""
    print(f"[Data] Đang đọc: {path}")
    df = pd.read_parquet(path)

    missing = [c for c in [src_col, tgt_col] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Không tìm thấy cột: {missing}\n"
            f"Các cột hiện có: {list(df.columns)}"
        )

    before = len(df)
    df = df.dropna(subset=[src_col, tgt_col])
    after = len(df)
    if before != after:
        print(f"[Data] Đã bỏ {before - after} dòng null. Còn lại: {after:,}")

    sources    = df[src_col].astype(str).tolist()
    references = df[tgt_col].astype(str).tolist()
    print(f"[Data] Đã load {len(sources):,} cặp câu.")
    return sources, references


# ---------------------------------------------------------------------------
# CTranslate2 inference
# ---------------------------------------------------------------------------

def translate_dataset(
    engine: TranslationEngine,
    sources: list[str],
    batch_size: int,
) -> list[str]:
    """
    Inference bằng CTranslate2 thông qua TranslationEngine:
    Gọi đúng engine.translate(src) cho từng câu để qua tầng _pack,
    đảm bảo tái hiện chính xác 100% luồng production.
    """
    total      = len(sources)
    hypotheses = []

    print(f"\n[CT2] Dịch {total:,} câu  "
          f"(batch={batch_size}, beam={engine.num_beams}, "
          f"max_input={engine.max_input_length}, max_target={engine.max_decode_length}) ...")

    t0 = time.time()
    for i, src in enumerate(sources):
        # Gọi đúng entry point của production
        translated = engine.translate(src)
        # engine.translate trả về list có 1 phần tử (đã join bằng \n nếu bị _pack tách)
        hypotheses.append(translated[0] if translated else "")

        done = i + 1
        # Vẫn dùng batch_size để log progress cho đỡ rác console
        if done % batch_size == 0 or done == total:
            elapsed = time.time() - t0
            speed   = done / elapsed
            eta     = (total - done) / speed if speed > 0 else 0
            print(f"  [{done:>6}/{total:,}]  {elapsed:6.1f}s  ETA {eta:5.1f}s  "
                  f"({speed:.1f} sent/s)")

    total_time = time.time() - t0
    print(f"[CT2] Hoàn tất {total:,} câu trong {total_time:.1f}s  "
          f"({total/total_time:.1f} sent/s)")
    return hypotheses


# ---------------------------------------------------------------------------
# COMET scoring
# ---------------------------------------------------------------------------

def run_comet(
    sources: list[str],
    hypotheses: list[str],
    references: list[str],
    comet_batch: int,
    device: str,
) -> object:
    """Download wmt22-comet-da và score."""
    print("\n[COMET] Load wmt22-comet-da ...")
    comet_path  = download_model("Unbabel/wmt22-comet-da")
    comet_model = load_from_checkpoint(comet_path)

    data = [
        {"src": s, "mt": h, "ref": r}
        for s, h, r in zip(sources, hypotheses, references)
    ]

    gpus = 1 if device == "cuda" else 0
    print(f"[COMET] Scoring {len(data):,} segments  (batch={comet_batch}, gpus={gpus}) ...")
    t0     = time.time()
    output = comet_model.predict(data, batch_size=comet_batch, gpus=gpus)
    print(f"[COMET] Hoàn tất trong {time.time() - t0:.1f}s")
    return output


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # Xác định device
    if args.device:
        device = args.device
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[Setup] Device: {device}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load data ────────────────────────────────────────────────────────
    sources, references = load_parquet(args.test_file, args.src_col, args.tgt_col)

    # ── 2. Load CT2 Translation Engine ──────────────────────────────────────
    print(f"[CT2] Load engine từ: {args.ct2_dir} và {args.tok_dir} (device={device})")
    engine = TranslationEngine(
        model_dir=args.ct2_dir,
        tokenizer_dir=args.tok_dir,
        device=device,
        max_input_length=args.max_input,
        num_beams=args.beam_size,
        max_decode_length=args.max_target,
        inter_threads=args.inter_threads
    )

    # ── 3. Translate ───────────────────────────────────────────
    hypotheses = translate_dataset(
        engine, sources,
        batch_size=args.batch_size,
    )

    # Lấy tên file test để làm prefix cho output
    file_stem = Path(args.test_file).stem

    if args.save_translations:
        hyp_path = out_dir / f"{file_stem}_hypotheses.txt"
        hyp_path.write_text("\n".join(hypotheses), encoding="utf-8")
        print(f"[Output] Bản dịch lưu tại: {hyp_path}")

    # ── 4. COMET scoring ────────────────────────────────────────────────────
    assert len(hypotheses) == len(sources), \
        f"Lỗi nghiêm trọng: Số lượng bản dịch ({len(hypotheses)}) khác với số lượng câu gốc ({len(sources)})."
    comet_output   = run_comet(sources, hypotheses, references, args.comet_batch, device)
    segment_scores = comet_output.scores
    system_score   = comet_output.system_score

    # ── 5. Lưu kết quả ──────────────────────────────────────────────────────
    detail_df = pd.DataFrame({
        "source":      sources,
        "hypothesis":  hypotheses,
        "reference":   references,
        "comet_score": segment_scores,
    }).sort_values("comet_score", ascending=True)  # worst → best để dễ phân tích lỗi

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

    # ── 6. In kết quả ───────────────────────────────────────────────────────
    print("\n" + "=" * 55)
    print("  COMET EVALUATION SUMMARY")
    print("=" * 55)
    print(f"  Metric        : wmt22-comet-da (reference-based)")
    print(f"  Segments      : {len(sources):,}")
    print(f"  System score  : {system_score:.4f}   ← con số chính để report")
    print(f"  Seg mean      : {summary['seg_mean']:.4f}")
    print(f"  Seg min/max   : {summary['seg_min']:.4f} / {summary['seg_max']:.4f}")
    print("=" * 55)
    print(f"\n  Kết quả lưu tại: {out_dir.resolve()}")
    print(f"    • {summary_path.name}  – điểm tổng thể")
    print(f"    • {detail_path.name}  – chi tiết từng câu (worst → best)")
    if args.save_translations:
        print(f"    • {hyp_path.name}      – bản dịch của model")
    print()


if __name__ == "__main__":
    main()
