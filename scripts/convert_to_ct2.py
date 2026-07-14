"""
Convert HuggingFace model (PyTorch) from Kaggle to CTranslate2 format.

Run ONCE after downloading and extracting the Kaggle output.
>> Not part of the app's runtime. <<

Usage:
    python scripts/convert_to_ct2.py \
        --model  path/to/final_model \
        --output models/en-vi-ct2 \
        --tokenizer-out models/tokenizer-en-vi

After running:
    models/en-vi-ct2/        <- CT2 model (model.bin, config.json, ...)
    models/tokenizer-en-vi/  <- Tokenizer files (source.spm, vocab, ...)
"""


import argparse
import shutil
import subprocess
import sys
from pathlib import Path


def convert(model_path, output_dir, tokenizer_out, quantization):
    model_path = Path(model_path)
    if not model_path.is_dir():
        sys.exit(f"Model directory not found: {model_path}")

    output_dir = Path(output_dir)
    if output_dir.exists():
        print(f"{output_dir} already exists -> deleting to convert again")
        shutil.rmtree(output_dir)

    # 1. Convert model to CTranslate2
    print(f"[1/2] Converting {model_path} -> {output_dir} (quant={quantization})")
    subprocess.run(
        [
            sys.executable, "-m", "ctranslate2.converters.transformers",
            "--model", str(model_path),
            "--output_dir", str(output_dir),
            "--quantization", quantization,
        ],
        check=True)

    # 2. Copy tokenizer files
    tokenizer_out = Path(tokenizer_out)
    tokenizer_out.mkdir(parents=True, exist_ok=True)
    tok_files = [
        "source.spm", "target.spm", "vocab.json",
        "tokenizer_config.json", "special_tokens_map.json",
        "config.json"
    ]
    print(f"[2/2] Copying tokenizer -> {tokenizer_out}")
    for name in tok_files:
        src = model_path / name
        if src.exists():
            shutil.copy(src, tokenizer_out / name)
            print(f"      + {name}")

    print("\n[OK] Done. The app can load from:")
    print(f"model     = {output_dir}")
    print(f"tokenizer = {tokenizer_out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="final_model directory from Kaggle")
    ap.add_argument("--output", default="models/en-vi-ct2", help="Output directory for CT2 model")
    ap.add_argument("--tokenizer-out", default="models/tokenizer-en-vi", help="Output directory for tokenizer")
    ap.add_argument("--quantization", default="int8", choices=["int8", "int8_float16", "float16", "float32"])
    args = ap.parse_args()
    convert(args.model, args.output, args.tokenizer_out, args.quantization)