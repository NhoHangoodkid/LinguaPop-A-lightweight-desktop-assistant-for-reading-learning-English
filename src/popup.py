"""
Popup nho hien ban dich gan con tro chuot.
Dung tkinter (co san trong Python). Chay tren luong chinh (main thread).

Vi tkinter khong an toan khi goi tu luong khac, hotkey listener se gui yeu cau
qua mot hang doi, va vong lap chinh trong main.py se goi show() o day.
"""
import re
import tkinter as tk


class Popup:
    def __init__(self, timeout_ms: int = 9000, font_size: int = 12, max_width: int = 460):
        self.timeout_ms = timeout_ms
        self.font_size = font_size
        self.max_width = max_width
        self.root = tk.Tk()
        self.root.withdraw()  # an cua so goc
        self._win = None
        self._timer_id = None
        self.on_history_click = None

    def _cancel_pending(self):
        if self._timer_id is not None:
            self.root.after_cancel(self._timer_id)
            self._timer_id = None

    def show(self, text, x, y):
        self._cancel_pending()
        
        # Dong popup cu neu con (destroy ngay lap tuc)
        if self._win is not None:
            try:
                self._win.destroy()
            except Exception:
                pass
            self._win = None

        win = tk.Toplevel(self.root)
        self._win = win
        win.overrideredirect(True)          # bo vien cua so
        win.attributes("-topmost", True)    # luon noi len tren
        win.attributes("-alpha", 0.0)       # bat dau voi alpha 0
        
        # Vien ngoai bang mau accent (Sky blue)
        win.configure(bg="#38bdf8")

        # Vi tri: gan con tro neu co, neu khong thi giua tren cung
        if x is None or y is None:
            x = win.winfo_screenwidth() // 2 - self.max_width // 2
            y = 80
        win.geometry(f"+{x + 12}+{y + 12}")

        # Nen chinh (Slate 900)
        container = tk.Frame(win, bg="#0f172a")
        container.pack(fill="both", expand=True, padx=1, pady=1)

        # Thanh accent bar mong o tren cung
        top_bar = tk.Frame(container, bg="#38bdf8", height=2)
        top_bar.pack(fill="x", side="top")

        frame = tk.Frame(container, bg="#0f172a", padx=16, pady=14)
        frame.pack(fill="both", expand=True)

        def make_copy_and_close_func(t):
            def _handler(e):
                self.root.clipboard_clear()
                self.root.clipboard_append(t)
                self.root.update()
                self._close()
                return "break"
            return _handler

        is_error_or_loading = text in ["Dang dich...", "(khong co ket qua)"] or text.startswith("Khong lay duoc") or text.startswith("Loi")
        parts = text.split("\n---\n")

        if len(parts) > 1 and not is_error_or_loading:
            for i, p in enumerate(parts):
                text_to_copy = p
                match = re.match(r"^\[\d+\]\s*(.*)$", p, flags=re.DOTALL)
                if match:
                    text_to_copy = match.group(1)
                
                opt_lbl = tk.Label(
                    frame, text=p, bg="#0f172a", fg="#f8fafc",
                    font=("Segoe UI", self.font_size), justify="left",
                    wraplength=self.max_width, anchor="w", cursor="hand2")
                opt_lbl.pack(fill="x", pady=(0, 4) if i < len(parts)-1 else 0)
                
                def _enter(e, l=opt_lbl): l.configure(bg="#1e293b")
                def _leave(e, l=opt_lbl): l.configure(bg="#0f172a")
                
                opt_lbl.bind("<Enter>", _enter)
                opt_lbl.bind("<Leave>", _leave)
                opt_lbl.bind("<Button-1>", make_copy_and_close_func(text_to_copy))
                
                if i < len(parts) - 1:
                    divider = tk.Frame(frame, bg="#334155", height=1)
                    divider.pack(fill="x", pady=4)
        else:
            label = tk.Label(
                frame, text=text, bg="#0f172a", fg="#f8fafc",
                font=("Segoe UI", self.font_size), justify="left",
                wraplength=self.max_width, anchor="w")
            
            if not is_error_or_loading:
                label.configure(cursor="hand2")
                def _enter(e): label.configure(bg="#1e293b")
                def _leave(e): label.configure(bg="#0f172a")
                label.bind("<Enter>", _enter)
                label.bind("<Leave>", _leave)
                label.bind("<Button-1>", make_copy_and_close_func(text))
                
            label.pack(fill="both", expand=True)

        # Duong phan cach mong truoc dong hint (Slate 700)
        sep = tk.Frame(frame, bg="#334155", height=1)
        sep.pack(fill="x", pady=(12, 8))

        bottom_bar = tk.Frame(frame, bg="#0f172a")
        bottom_bar.pack(fill="x")

        if self.on_history_click:
            hist_lbl = tk.Label(
                bottom_bar, text="Lịch sử dịch", bg="#0f172a", fg="#38bdf8",
                font=("Segoe UI", 8, "underline"), cursor="hand2")
            hist_lbl.pack(side="left")
            def _on_hist(e):
                self.on_history_click()
            hist_lbl.bind("<Button-1>", _on_hist)

        hint = tk.Label(
            bottom_bar, text="Esc · click de dong", bg="#0f172a", fg="#94a3b8",
            font=("Segoe UI", 8))
        hint.pack(side="right")

        # Dong bang Esc hoac click (gan cho cac thanh phan)
        def close_handler(e):
            self._close()
            
        win.bind("<Escape>", close_handler)
        win.bind("<Button-1>", close_handler)
        hint.bind("<Button-1>", close_handler)
        bottom_bar.bind("<Button-1>", close_handler)
        frame.bind("<Button-1>", close_handler)
        
        win.focus_force()

        self._fade_in(win, 0.0)

        if self.timeout_ms and self.timeout_ms > 0:
            self._timer_id = self.root.after(self.timeout_ms, self._close)

    def _fade_in(self, win, alpha):
        if win != self._win or not win.winfo_exists():
            return
        alpha += 0.08
        if alpha >= 0.95:
            win.attributes("-alpha", 0.95)
        else:
            win.attributes("-alpha", alpha)
            self.root.after(15, lambda: self._fade_in(win, alpha))

    def _fade_out(self, win, alpha):
        if not win.winfo_exists():
            return
        alpha -= 0.08
        if alpha <= 0:
            win.destroy()
        else:
            win.attributes("-alpha", alpha)
            self.root.after(15, lambda: self._fade_out(win, alpha))

    def show_loading(self, x, y):
        self.show("Dang dich...", x, y)

    def _close(self):
        self._cancel_pending()
        if self._win is not None:
            win_to_close = self._win
            self._win = None
            self._fade_out(win_to_close, 0.95)

    def run_mainloop(self):
        self.root.mainloop()