"""
Translation history viewer window. Opens via hotkey, with search and copy.
"""
import re
import tkinter as tk
from tkinter import ttk, messagebox
from history import TranslationHistory


class HistoryWindow:
    def __init__(self, history: TranslationHistory):
        self.history = history
        self._win = None

    def is_open(self) -> bool:
        return self._win is not None and self._win.winfo_exists()

    def toggle(self, root: tk.Tk):
        """Open if closed, close if open."""
        if self.is_open():
            self._win.destroy()
            self._win = None
            return
        self._open(root)

    def _open(self, root: tk.Tk):
        win = tk.Toplevel(root)
        self._win = win
        win.title("Translation History")
        win.geometry("720x500")
        win.configure(bg="#1a1b2e")
        win.attributes("-topmost", True)

        # -- Colors --
        BG = "#1a1b2e"
        FG = "#e8eaf6"
        ACCENT = "#7c4dff"
        ENTRY_BG = "#2d2f4a"

        # -- Search bar --
        search_frame = tk.Frame(win, bg=BG, pady=8, padx=12)
        search_frame.pack(fill="x")

        tk.Label(search_frame, text="\U0001f50d", bg=BG, fg=FG,
                 font=("Segoe UI", 12)).pack(side="left")

        self.search_var = tk.StringVar()
        search_entry = tk.Entry(
            search_frame, textvariable=self.search_var,
            bg=ENTRY_BG, fg=FG, insertbackground=FG,
            font=("Segoe UI", 11), relief="flat", bd=0)
        search_entry.pack(side="left", fill="x", expand=True, padx=(8, 8), ipady=4)
        search_entry.focus()

        # Search on keypress (debounce 300ms)
        self._search_after = None
        def on_search_change(*_):
            if self._search_after:
                win.after_cancel(self._search_after)
            self._search_after = win.after(300, self._refresh)
        self.search_var.trace_add("write", on_search_change)

        # Clear all button
        btn_clear = tk.Button(
            search_frame, text="Clear All", bg="#c62828", fg="white",
            font=("Segoe UI", 9), relief="flat", cursor="hand2",
            command=self._clear_all)
        btn_clear.pack(side="right")

        # -- Record count --
        self.count_label = tk.Label(
            win, text="", bg=BG, fg="#6c6f85",
            font=("Segoe UI", 9), anchor="w")
        self.count_label.pack(fill="x", padx=14)

        # -- History list (scrollable) --
        container = tk.Frame(win, bg=BG)
        container.pack(fill="both", expand=True, padx=8, pady=(4, 8))

        canvas = tk.Canvas(container, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(container, orient="vertical", command=canvas.yview)
        self.list_frame = tk.Frame(canvas, bg=BG)

        self.list_frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")))
        canvas.create_window((0, 0), window=self.list_frame, anchor="nw")
        canvas.configure(yscrollcommand=scrollbar.set)

        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        # Mouse wheel scrolling
        def on_mousewheel(e):
            if canvas.winfo_exists():
                canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")
        win.bind("<MouseWheel>", on_mousewheel)

        # Keyboard shortcuts
        win.bind("<Escape>", lambda e: win.destroy())

        self._canvas = canvas
        self._refresh()

    def _refresh(self):
        """Reload list from DB."""
        for w in self.list_frame.winfo_children():
            w.destroy()

        keyword = self.search_var.get().strip()
        if keyword:
            records = self.history.search(keyword, limit=100)
        else:
            records = self.history.recent(limit=100)

        total = self.history.count()
        showing = len(records)
        self.count_label.config(
            text=f"  Showing {showing} / {total} records")

        BG = "#1a1b2e"
        CARD_BG = "#252742"
        FG = "#e8eaf6"
        SRC_FG = "#b0b3cc"
        ACCENT = "#7c4dff"
        TIME_FG = "#6c6f85"

        if not records:
            tk.Label(self.list_frame, text="No history yet.",
                     bg=BG, fg=TIME_FG, font=("Segoe UI", 11)).pack(pady=40)
            return

        for rec in records:
            card = tk.Frame(self.list_frame, bg=CARD_BG, pady=8, padx=12)
            card.pack(fill="x", pady=3, padx=4)

            # Row 1: timestamp + method
            header = tk.Frame(card, bg=CARD_BG)
            header.pack(fill="x")

            method_text = "\U0001f4cb Selection" if rec["method"] == "selection" else "\U0001f4f7 OCR"
            tk.Label(header, text=method_text, bg=CARD_BG, fg=ACCENT,
                     font=("Segoe UI", 8, "bold")).pack(side="left")
            tk.Label(header, text=rec["created_at"], bg=CARD_BG, fg=TIME_FG,
                     font=("Segoe UI", 8)).pack(side="right")

            # Row 2: source text (truncated)
            src = rec["source_text"]
            if len(src) > 120:
                src = src[:120] + "\u2026"
            tk.Label(card, text=src, bg=CARD_BG, fg=SRC_FG,
                     font=("Segoe UI", 9), anchor="w", justify="left",
                     wraplength=650).pack(fill="x", pady=(4, 2))

            # Row 3: translation
            trans = rec["translated"]
            if len(trans) > 200:
                trans = trans[:200] + "\u2026"
            tk.Label(card, text=trans, bg=CARD_BG, fg=FG,
                     font=("Segoe UI", 10), anchor="w", justify="left",
                     wraplength=650).pack(fill="x")

            # Copy + delete buttons
            btn_frame = tk.Frame(card, bg=CARD_BG)
            btn_frame.pack(fill="x", pady=(4, 0))

            rid = rec["id"]
            full_trans = rec["translated"]

            parts = full_trans.split("\n---\n")
            if len(parts) > 1:
                for i, p in enumerate(parts):
                    text_to_copy = p
                    match = re.match(r"^\[\d+\]\s*(.*)$", p, flags=re.DOTALL)
                    if match:
                        text_to_copy = match.group(1)
                        
                    copy_btn = tk.Label(
                        btn_frame, text=f"\U0001f4cb Copy [{i+1}]", bg=CARD_BG, fg=ACCENT,
                        font=("Segoe UI", 8), cursor="hand2")
                    copy_btn.pack(side="left", padx=(0, 8) if i < len(parts)-1 else 0)
                    copy_btn.bind("<Button-1>",
                                  lambda e, t=text_to_copy: self._copy(t))
            else:
                copy_btn = tk.Label(
                    btn_frame, text="\U0001f4cb Copy", bg=CARD_BG, fg=ACCENT,
                    font=("Segoe UI", 8), cursor="hand2")
                copy_btn.pack(side="left")
                copy_btn.bind("<Button-1>",
                              lambda e, t=full_trans: self._copy(t))

            del_btn = tk.Label(
                btn_frame, text="\U0001f5d1 Delete", bg=CARD_BG, fg="#c62828",
                font=("Segoe UI", 8), cursor="hand2")
            del_btn.pack(side="left", padx=(12, 0))
            del_btn.bind("<Button-1>",
                         lambda e, r=rid: self._delete(r))

    def _copy(self, text: str):
        if self._win:
            self._win.clipboard_clear()
            self._win.clipboard_append(text)

    def _delete(self, record_id: int):
        self.history.delete(record_id)
        self._refresh()

    def _clear_all(self):
        if self._win:
            ok = messagebox.askyesno(
                "Confirm", "Delete all translation history?",
                parent=self._win)
            if ok:
                self.history.clear_all()
                self._refresh()
