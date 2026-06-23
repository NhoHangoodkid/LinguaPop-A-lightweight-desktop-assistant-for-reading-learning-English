"""
Lay text dang boi den: backup clipboard -> gia lap Ctrl+C -> doc clipboard
-> tra clipboard cu lai. Co tach cau cho doan dai.

OCR sau nay: them ham capture_from_image() vao day, tra ve cung kieu str.
"""
import re
import time

import pyperclip
from pynput.keyboard import Controller, Key

_kbd = Controller()

# Mot cau ket thuc bang . ! ? (co the kem dau dong/ngoac), theo sau la khoang trang.
# Dung de tach cau MA VAN giu nguyen dau cau o cuoi moi cau.
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])["\')\]]?\s+')


def get_selected_text() -> str:
    """Tra ve text dang boi den, hoac chuoi rong neu khong lay duoc."""
    backup = ""
    try:
        backup = pyperclip.paste()
    except Exception:
        pass

    try:
        pyperclip.copy("")
    except Exception:
        pass

    for k in [Key.shift, Key.shift_r, Key.alt, Key.alt_gr, Key.alt_r,
              Key.cmd, Key.cmd_r, Key.ctrl, Key.ctrl_r]:
        _kbd.release(k)

    time.sleep(0.05)

    _kbd.press(Key.ctrl)
    _kbd.press("c")
    _kbd.release("c")
    _kbd.release(Key.ctrl)

    time.sleep(0.15)

    try:
        selected = pyperclip.paste()
    except Exception:
        selected = ""

    try:
        pyperclip.copy(backup)
    except Exception:
        pass

    return selected.strip()


def split_sentences(text: str) -> list:
    """
    Tach doan thanh danh sach cac khoi (block) de giu nguyen context (khoang trang, xuong dong).
    Moi khoi la mot dict: {"text": "cau hoan chinh", "delimiter": "khoang trang / xuong dong phia sau"}
    """
    text = text.strip()
    if not text:
        return []

    # Fix loi cat tu: neu co dau gach ngang o cuoi dong (vd: dimen-\nsion), noi chung lai voi nhau
    text = re.sub(r'-[ \t]*\n[ \t]*', '', text)

    blocks = []
    last_end = 0
    # Dung finditer de tim cac ranh gioi cau (dau cau + khoang trang/xuong dong)
    for match in _SENT_BOUNDARY.finditer(text):
        start, end = match.span()
        sentence = text[last_end:start]
        delimiter = text[start:end]
        
        # Neu cau rong, cong don delimiter vao khoi truoc do
        if not sentence.strip() and blocks:
            blocks[-1]["delimiter"] += sentence + delimiter
        else:
            blocks.append({"text": sentence, "delimiter": delimiter})
        last_end = end
        
    # Phan con lai cuoi cung (neu co)
    remaining = text[last_end:]
    if remaining:
        if not remaining.strip() and blocks:
            blocks[-1]["delimiter"] += remaining
        else:
            blocks.append({"text": remaining, "delimiter": ""})
            
    return blocks