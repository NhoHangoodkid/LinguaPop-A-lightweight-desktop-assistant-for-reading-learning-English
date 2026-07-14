"""
Screen region capture -> OCR with RapidOCR -> return text.

RapidOCR is loaded ONCE (like TranslationEngine),
each call only runs inference -> much faster than reinitializing.
"""
import tkinter as tk
import numpy as np
import cv2
from PIL import ImageGrab, ImageTk, ImageEnhance
from rapidocr_onnxruntime import RapidOCR


class OCREngine:
    """Load RapidOCR once, reuse throughout the app."""

    def __init__(self, lang: str = "en", use_gpu: bool = False):
        print(f"[OCR] Loading RapidOCR ...")
        self.engine = RapidOCR()
        print("[OCR] Ready.")

    def recognize(self, img) -> str:
        """Accept PIL Image, return recognized text."""
        # Convert PIL Image (RGB) to BGR for RapidOCR (OpenCV format)
        img_array = np.array(img)
        img_bgr = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
        
        result, _ = self.engine(img_bgr)

        if not result:
            return ""

        # RapidOCR result format: list of tuples (box, text, score)
        # box is a list of 4 coordinate points
        lines = []
        for bbox, text, conf in result:
            top_y = min(p[1] for p in bbox)
            left_x = min(p[0] for p in bbox)
            lines.append((top_y, left_x, text))

        # Group into rows: boxes with close top_y (<15px) = same row
        lines.sort(key=lambda x: (x[0], x[1]))
        grouped = []
        current_row = [lines[0]]
        for item in lines[1:]:
            if abs(item[0] - current_row[0][0]) < 15:
                current_row.append(item)
            else:
                current_row.sort(key=lambda x: x[1])  # sort left->right
                grouped.append(" ".join(t for _, _, t in current_row))
                current_row = [item]
        current_row.sort(key=lambda x: x[1])
        grouped.append(" ".join(t for _, _, t in current_row))

        return "\n".join(grouped)


class ScreenCapture:
    """Let user drag-select a region on screen."""

    def __init__(self):
        self._result = None
        self._center = (0, 0)

    def grab(self):
        """Return (PIL Image, center_x, center_y) or (None, 0, 0)."""
        self._result = None
        screenshot = ImageGrab.grab()

        root = tk.Toplevel()
        root.attributes("-fullscreen", True)
        root.attributes("-topmost", True)
        root.configure(cursor="crosshair")

        dimmed = ImageEnhance.Brightness(screenshot).enhance(0.6)
        tk_img = ImageTk.PhotoImage(dimmed)

        canvas = tk.Canvas(root, highlightthickness=0)
        canvas.pack(fill="both", expand=True)
        canvas.create_image(0, 0, anchor="nw", image=tk_img)
        canvas.image = tk_img  # Keep reference to prevent garbage collection

        rect_id = None
        start = [0, 0]

        def on_press(e):
            nonlocal rect_id
            start[0], start[1] = e.x, e.y
            rect_id = canvas.create_rectangle(
                e.x, e.y, e.x, e.y,
                outline="#7c4dff", width=2, dash=(6, 4))

        def on_drag(e):
            if rect_id:
                canvas.coords(rect_id, start[0], start[1], e.x, e.y)

        def on_release(e):
            x1, y1 = min(start[0], e.x), min(start[1], e.y)
            x2, y2 = max(start[0], e.x), max(start[1], e.y)
            if (x2 - x1) > 10 and (y2 - y1) > 10:
                self._result = screenshot.crop((x1, y1, x2, y2))
                self._center = ((x1 + x2) // 2, (y1 + y2) // 2)
            root.destroy()

        def on_escape(e):
            root.destroy()

        canvas.bind("<ButtonPress-1>", on_press)
        canvas.bind("<B1-Motion>", on_drag)
        canvas.bind("<ButtonRelease-1>", on_release)
        root.bind("<Escape>", on_escape)

        canvas.create_text(
            screenshot.width // 2, 40,
            text="Drag to select region to translate - Esc to cancel",
            fill="#e8eaf6", font=("Segoe UI", 14, "bold"))

        root.wait_window(root)
        return self._result, self._center


def capture_and_ocr(ocr_engine: OCREngine) -> tuple:
    """
    Capture screen -> OCR -> return (text, x, y).
    x, y is the center of the selected region for popup positioning.
    """
    cap = ScreenCapture()
    img, (cx, cy) = cap.grab()
    if img is None:
        return None, 0, 0

    text = ocr_engine.recognize(img)
    return (text if text else None), cx, cy
