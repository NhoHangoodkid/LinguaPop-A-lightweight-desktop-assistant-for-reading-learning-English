"""
Shared utility functions for the data pipeline.
Includes: text cleaning, quality filtering, dedup, LaBSE similarity.
"""

import re
import unicodedata
import numpy as np
import pandas as pd
from collections import Counter


# Text cleaning

_re_url = re.compile(r"http\S+|www\S+|<.*?>")
_re_ws  = re.compile(r"\s+")


def clean_series(s: pd.Series) -> pd.Series:
    """NFC normalize, remove URLs, clean whitespace."""
    s = s.fillna("").astype(str)
    s = s.map(lambda x: unicodedata.normalize("NFC", x))
    s = s.str.replace(_re_url, "", regex=True)
    s = s.str.replace(_re_ws, " ", regex=True).str.strip()
    return s


def read_file(path: str) -> str:
    """Read text file with encoding fallback."""
    for enc in ["utf-8", "utf-8-sig", "latin-1"]:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue
    return ""


# Quality filtering

def digit_signature(text: str) -> list:
    """Extract digit groups from text."""
    return re.findall(r"\d+", text)


def repeated_token(text: str) -> bool:
    """Detect excessive token repetition."""
    words = text.split()
    if len(words) < 10:
        return False
    return Counter(words).most_common(1)[0][1] / len(words) > 0.4


def _is_latin_vn(c: str) -> bool:
    o = ord(c)
    return (o < 128 or 0x00C0 <= o <= 0x00FF or 0x0100 <= o <= 0x024F
            or 0x0300 <= o <= 0x036F or 0x1EA0 <= o <= 0x1EF9)


def non_latin_ratio(text: str) -> float:
    """Ratio of non-Latin-Vietnamese characters in text."""
    if not text:
        return 0.0
    return sum(0 if _is_latin_vn(c) else 1 for c in text) / len(text)


_re_excel  = re.compile(r"#(NAME\?|REF!|VALUE!|DIV/0!|N/A|NUM!|NULL!)")
_re_entity = re.compile(r"#\s*\d{2,};|&#?\w+;")
_re_holder = re.compile(r"%\s*\d+|%[sd]\b|\{\d+")
_vi_markers = ["đ", "ă", "â", "ê", "ô", "ơ", "ư"]


def is_bad_pair(en: str, vi: str) -> bool:
    """Identify low-quality sentence pairs based on multiple criteria."""
    txt = en + " " + vi

    if txt.count("<unk>") >= 1:
        return True
    if re.search(r"\.{8,}", txt) or re.search(r"[._-]{10,}", txt):
        return True
    if any(x in txt for x in ["Ã", "Â", "â€", "Ð", "Ñ", "\ufffd"]):
        return True
    if _re_excel.search(txt) or _re_entity.search(txt) or _re_holder.search(txt):
        return True
    if repeated_token(txt):
        return True
    words = txt.split()
    if len(words) >= 10 and len(set(words)) / len(words) < 0.4:
        return True
    if txt.count(".") > 30:
        return True
    if non_latin_ratio(en) > 0.30 or non_latin_ratio(vi) > 0.30:
        return True
    if sum(ch in en.lower() for ch in _vi_markers) >= 2:
        return True
    if en.strip().lower() == vi.strip().lower():
        return True
    en_nums, vi_nums = digit_signature(en), digit_signature(vi)
    if en_nums and vi_nums and en_nums != vi_nums:
        return True
    return False


def filter_quality(df: pd.DataFrame) -> pd.DataFrame:
    """Filter sentence pairs by quality criteria: word count, length ratio."""
    vl = np.nan_to_num(df["vi"].str.split().str.len().to_numpy()).astype(int)
    el = np.nan_to_num(df["en"].str.split().str.len().to_numpy()).astype(int)
    hi = np.maximum(vl, el)
    lo = np.minimum(vl, el).clip(min=1)
    mask = (
        (df["vi"].str.len().to_numpy() > 0) &
        (df["en"].str.len().to_numpy() > 0) &
        (vl >= 3) & (vl <= 80) &
        (el >= 3) & (el <= 80) &
        (hi / lo <= 2.0)
    )
    return df[mask].copy()


# Dedup & Language filter

def dedup(df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicates by en and vi columns separately."""
    df = df.drop_duplicates(subset=["en", "vi"])
    df = df.drop_duplicates(subset=["en"])
    df = df.drop_duplicates(subset=["vi"])
    return df.reset_index(drop=True)


def apply_lang_filter(df: pd.DataFrame, use_fasttext: bool = False,
                      fasttext_path: str = "lid.176.bin") -> pd.DataFrame:
    """Optional language filtering with FastText (disabled by default)."""
    if not use_fasttext:
        return df
    import fasttext
    lid = fasttext.load_model(fasttext_path)

    def ok(en, vi):
        le, pe = lid.predict(en.replace("\n", " "))
        lv, pv = lid.predict(vi.replace("\n", " "))
        return (le[0] == "__label__en" and pe[0] > 0.5 and
                lv[0] == "__label__vi" and pv[0] > 0.5)

    keep = [ok(en, vi) for en, vi in zip(df["en"], df["vi"])]
    return df[keep].reset_index(drop=True)


# LaBSE Cross-Lingual Similarity

def load_labse(device: str = None):
    """Load LaBSE model onto GPU (FP16) or CPU."""
    import torch
    from sentence_transformers import SentenceTransformer

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SentenceTransformer("sentence-transformers/LaBSE", device=device)
    if device == "cuda":
        model = model.half()
    print(f"[LaBSE] Loaded on {device}" + (" (FP16)" if device == "cuda" else ""))
    return model


def add_cross_sim(df: pd.DataFrame, labse_model,
                  batch_size: int = 128) -> pd.DataFrame:
    """Compute cross-lingual cosine similarity using LaBSE."""
    e = labse_model.encode(
        df["en"].tolist(), batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=True
    )
    v = labse_model.encode(
        df["vi"].tolist(), batch_size=batch_size,
        normalize_embeddings=True, show_progress_bar=True
    )
    df = df.copy()
    sim = (np.asarray(e) * np.asarray(v)).sum(1)
    df["sim"] = np.clip(sim, -1.0, 1.0)
    return df


def remove_overlap(train_df: pd.DataFrame, *holdouts) -> pd.DataFrame:
    """Remove overlap between train and holdout sets (valid/test)."""
    h_en, h_vi = set(), set()
    for h in holdouts:
        h_en |= set(h["en"])
        h_vi |= set(h["vi"])
    before = len(train_df)
    out = train_df[
        ~train_df["en"].isin(h_en) & ~train_df["vi"].isin(h_vi)
    ].reset_index(drop=True)
    print(f"  Remove overlap: {before:,} -> {len(out):,}")
    return out
