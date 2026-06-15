"""
Entry point cua app dich.

Luong hoat dong:
  1. Doc config, load engine (CTranslate2) MOT LAN
  2. Dang ky hotkey toan cuc tren luong rieng
  3. Khi nhan hotkey -> day su kien vao hang doi
  4. Vong lap tkinter o luong chinh kiem tra hang doi -> lay text -> dich -> popup

Vi sao co hang doi: tkinter chi an toan khi goi tu luong chinh, con pynput
goi callback tu luong rieng. Hang doi la cau noi an toan giua hai luong.
"""
import queue
import sys
import ctypes

try:
    # Cố gắng set DPI awareness để tránh bị zoom màn hình khi Windows scale > 100%
    ctypes.windll.shcore.SetProcessDpiAwareness(2) # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

from pathlib import Path

import yaml
from pynput.mouse import Controller as MouseController

# Cho phep import module trong cung thu muc src/
sys.path.insert(0, str(Path(__file__).parent))

from engine import TranslationEngine, EngineManager   
from capture import get_selected_text, split_sentences  
from popup import Popup                                  
from hotkey import HotkeyListener                       
from history import TranslationHistory
from history_window import HistoryWindow
import ocr

ROOT = Path(__file__).parent.parent


def load_config():
    with open(ROOT / "config.yaml", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_engine_manager(cfg) -> EngineManager:
    mgr = EngineManager()
    t = cfg["translation"]
    default_name = cfg["engines"]["default"]

    for name, spec in cfg["engines"].items():
        if name == "default":
            continue
        model_dir = ROOT / spec["model_dir"]
        tok_dir = ROOT / spec["tokenizer_dir"]
        if not model_dir.is_dir():
            print(f"Bo qua engine '{name}': khong thay {model_dir}")
            continue
        print(f"Load engine '{name}' tu {model_dir} ...")
        eng = TranslationEngine(
            model_dir=str(model_dir),
            tokenizer_dir=str(tok_dir),
            device=cfg["device"],
            max_input_length=t["max_input_length"],
            num_beams=t["num_beams"],
            num_hypotheses=t.get("num_hypotheses", 1),
            max_decode_length=t["max_decode_length"])
        mgr.register(name, eng, is_default=(name == default_name))
    return mgr


def main():
    cfg = load_config()

    mgr = build_engine_manager(cfg)
    if mgr._default is None:
        sys.exit("Khong load duoc engine nao. Kiem tra thu muc models/ "
                 "va da chay convert_to_ct2.py chua.")

    ui = cfg["ui"]
    popup = Popup(timeout_ms=ui["popup_timeout_ms"], font_size=ui["font_size"], max_width=ui["max_width"])
    mouse = MouseController()
    
    history_db = TranslationHistory()
    history_win = HistoryWindow(history_db)
    
    ocr_lang = "vi" if "vie" in cfg["ocr"].get("lang", "").lower() else "en"
    ocr_engine_inst = ocr.OCREngine(lang=ocr_lang, use_gpu=(cfg.get("device") == "cuda"))

    # Hang doi noi luong hotkey -> luong chinh (tkinter)
    events = queue.Queue()

    def on_hotkey_sel():
        events.put("translate_sel")

    def on_hotkey_ocr():
        events.put("translate_ocr")

    listener_sel = HotkeyListener(cfg["hotkeys"]["translate_selection"], on_hotkey_sel)
    listener_sel.start()
    
    listener_ocr = HotkeyListener(cfg["hotkeys"]["translate_ocr"], on_hotkey_ocr)
    listener_ocr.start()

    popup.on_history_click = lambda: events.put("show_history")

    print(f"\n[OK] San sang.")
    print(f" - Boi den text + nhan {cfg['hotkeys']['translate_selection'].upper()} de dich.")
    print(f" - Nhan {cfg['hotkeys']['translate_ocr'].upper()} de chup man hinh va dich (OCR).")
    print("Dong cua so terminal nay de thoat app.\n")

    is_translating = False

    def process_queue():
        last_event = None
        try:
            while True:
                last_event = events.get_nowait()
        except queue.Empty:
            pass
            
        if last_event == "translate_sel":
            handle_translate_sel()
        elif last_event == "translate_ocr":
            handle_translate_ocr()
        elif last_event == "show_history":
            history_win.toggle(popup.root)
            
        # Kiem tra lai sau 80ms
        popup.root.after(80, process_queue)

    def handle_translate_sel():
        nonlocal is_translating
        if is_translating:
            return
            
        is_translating = True
        try:
            text = get_selected_text()
            x, y = mouse.position
            if not text:
                popup.show("Khong lay duoc text boi den.", x, y)
                return
            popup.show_loading(x, y)
            popup.root.update_idletasks()
            
            sentences = split_sentences(text)
            results = mgr.translate(sentences)
            
            if not results:
                final_text = "(khong co ket qua)"
            elif len(results) == 1:
                final_text = results[0]
            else:
                final_text = "\n---\n".join(f"[{i+1}] {r}" for i, r in enumerate(results))
                
            popup.show(final_text, x, y)
            if results:
                history_db.save(text, final_text, method="selection")
        except Exception as e:
            popup.show(f"Loi dich: {e}", x, y)
        finally:
            is_translating = False

    def handle_translate_ocr():
        nonlocal is_translating
        if is_translating:
            return
            
        is_translating = True
        x, y = None, None  # Khoi tao x, y de phong loi UnboundLocalError
        try:
            # Goi ocr
            text, x_new, y_new = ocr.capture_and_ocr(ocr_engine_inst)
            if text is None:
                # Nguoi dung huy hoac loi
                return
            x, y = x_new, y_new
            if not text:
                popup.show("Khong nhan dien duoc chu nao trong vung chon.", x, y)
                return
                
            popup.show_loading(x, y)
            popup.root.update_idletasks()
            
            sentences = split_sentences(text)
            results = mgr.translate(sentences)
            
            if not results:
                final_text = "(khong co ket qua)"
            elif len(results) == 1:
                final_text = results[0]
            else:
                final_text = "\n---\n".join(f"[{i+1}] {r}" for i, r in enumerate(results))
                
            # Chi hien thi ban dich
            popup.show(final_text, x, y)
            if results:
                history_db.save(text, final_text, method="ocr")
            
        except Exception as e:
            popup.show(f"Loi OCR/Dich: {e}", x, y)
        finally:
            is_translating = False

    popup.root.after(80, process_queue)
    try:
        popup.run_mainloop()
    finally:
        listener_sel.stop()
        listener_ocr.stop()


if __name__ == "__main__":
    main()