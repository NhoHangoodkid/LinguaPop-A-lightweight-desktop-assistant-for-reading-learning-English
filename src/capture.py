"""
Get selected text: backup clipboard -> simulate Ctrl+C -> read clipboard
-> restore old clipboard. Includes sentence splitting for long paragraphs.

OCR later: add a capture_from_image() function here, returning the same str type.
"""
import re
import time

import pyperclip
from pynput.keyboard import Controller, Key

_kbd = Controller()

# A sentence ends with . ! ? (optionally followed by closing quotes/brackets), followed by whitespace.
# Used to split sentences WHILE keeping the punctuation at the end of each sentence.
_SENT_BOUNDARY = re.compile(r'(?<=[.!?])["\')\]]?\s+')


def get_selected_text() -> str:
    """Returns the currently selected text, or an empty string if unable to get it."""
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
    Split a paragraph into a list of blocks to preserve context (whitespace, newlines).
    Each block is a dict: {"text": "complete sentence", "delimiter": "whitespace / newline after it"}
    """
    text = text.strip()
    if not text:
        return []

    # Fix word wrap: if there is a hyphen at the end of a line (e.g., dimen-\nsion), join them together
    text = re.sub(r'-[ \t]*\n[ \t]*', '', text)

    blocks = []
    last_end = 0
    # Use finditer to find sentence boundaries (punctuation + whitespace/newline)
    for match in _SENT_BOUNDARY.finditer(text):
        start, end = match.span()
        sentence = text[last_end:start]
        delimiter = text[start:end]
        
        # If the sentence is empty, append the delimiter to the previous block
        if not sentence.strip() and blocks:
            blocks[-1]["delimiter"] += sentence + delimiter
        else:
            blocks.append({"text": sentence, "delimiter": delimiter})
        last_end = end
        
        # The remaining part at the end (if any)
    remaining = text[last_end:]
    if remaining:
        if not remaining.strip() and blocks:
            blocks[-1]["delimiter"] += remaining
        else:
            blocks.append({"text": remaining, "delimiter": ""})
            
    return blocks