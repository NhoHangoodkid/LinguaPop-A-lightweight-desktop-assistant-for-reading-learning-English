"""
Entry point of the translation app.

Workflow:
  1. Read config, load engine (CTranslate2) ONCE
  2. Register global hotkeys on a separate thread
  3. When hotkey pressed -> push event into queue
  4. Tkinter main loop on main thread checks queue -> get text -> translate -> popup

Why use a queue: tkinter is only safe when called from the main thread, while pynput
calls callbacks from a separate thread. The queue bridges the two threads safely.
"""
import queue
import sys
import ctypes
import threading

try:
    # Set DPI awareness to avoid zoom issues when Windows scale > 100%
    ctypes.windll.shcore.SetProcessDpiAwareness(2) # PROCESS_PER_MONITOR_DPI_AWARE
except Exception:
    try:
        ctypes.windll.user32.SetProcessDPIAware()
    except Exception:
        pass

from pathlib import Path

import yaml
from pynput.mouse import Controller as MouseController

# Allow importing modules from the same src/ directory
sys.path.insert(0, str(Path(__file__).parent))

from engine import TranslationEngine, EngineManager   
from capture import get_selected_text, split_sentences  
from popup import Popup                                  
from hotkey import HotkeyListener                       
from history import TranslationHistory
from history_window import HistoryWindow
from tts import TTSEngine
from pronunciation import PronunciationEvaluator
from pronunciation.recorder import record_audio
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
            print(f"Skipping engine '{name}': not found {model_dir}")
            continue
        print(f"Loading engine '{name}' from {model_dir} ...")
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
        sys.exit("Could not load any engine. Check the models/ directory "
                 "and ensure convert_to_ct2.py has been run.")

    ui = cfg["ui"]
    popup = Popup(timeout_ms=ui["popup_timeout_ms"], font_size=ui["font_size"], max_width=ui["max_width"])
    mouse = MouseController()
    
    history_db = TranslationHistory()
    history_win = HistoryWindow(history_db)

    # Initialize TTS Engine
    tts_cfg = cfg.get("tts", {})
    tts_engine = TTSEngine(tts_cfg)
    pronunciation_cfg = cfg.get("pronunciation", {})
    pronunciation_enabled = pronunciation_cfg.get("enabled", False)
    if pronunciation_enabled and pronunciation_cfg.get("sample_rate", 16000) != 16000:
        raise ValueError("The pronunciation model requires a 16000 Hz sample rate.")
    pronunciation_evaluator = PronunciationEvaluator(
        model_size=pronunciation_cfg.get("model_size", "tiny"),
        device=pronunciation_cfg.get("device", cfg.get("device", "cpu")),
        compute_type=pronunciation_cfg.get("compute_type", "int8"),
        language=pronunciation_cfg.get("language", "en"),
    )
    events = queue.Queue()

    # Connect TTS callbacks
    popup.on_speak = lambda text, lang: tts_engine.speak_async(text, lang)
    popup.on_stop_speak = lambda: tts_engine.stop()
    if pronunciation_enabled:
        popup.on_pronunciation = lambda text: events.put(("check_pronunciation", text))
    history_win.on_speak = lambda text, lang: tts_engine.speak_async(text, lang)
    
    ocr_lang = "vi" if "vie" in cfg["ocr"].get("lang", "").lower() else "en"
    ocr_engine_inst = ocr.OCREngine(lang=ocr_lang, use_gpu=(cfg.get("device") == "cuda"))

    # Queue bridging hotkey thread -> main thread (tkinter)

    def on_hotkey_sel():
        events.put("translate_sel")

    def on_hotkey_ocr():
        events.put("translate_ocr")

    def on_hotkey_speak():
        events.put("speak_sel")

    listener_sel = HotkeyListener(cfg["hotkeys"]["translate_selection"], on_hotkey_sel)
    listener_sel.start()
    
    listener_ocr = HotkeyListener(cfg["hotkeys"]["translate_ocr"], on_hotkey_ocr)
    listener_ocr.start()

    listener_speak = None
    if "speak_selection" in cfg.get("hotkeys", {}) and cfg["hotkeys"]["speak_selection"]:
        listener_speak = HotkeyListener(cfg["hotkeys"]["speak_selection"], on_hotkey_speak)
        listener_speak.start()

    popup.on_history_click = lambda: events.put("show_history")

    print(f"\n[OK] Ready.")
    print(f" - Select text + press {cfg['hotkeys']['translate_selection'].upper()} to translate.")
    print(f" - Press {cfg['hotkeys']['translate_ocr'].upper()} to capture screen and translate (OCR).")
    if listener_speak:
        print(f" - Select text + press {cfg['hotkeys']['speak_selection'].upper()} to pronounce (read aloud).")
    if tts_engine.enabled:
        print(f" - TTS Engine active ({tts_engine.mode}). Press [E] for English audio, [V] for Vietnamese audio on popup.")
    print("Close this terminal window to exit the app.\n")

    is_translating = False
    is_checking_pronunciation = False

    def process_queue():
        last_event = None
        pronunciation_events = []
        try:
            while True:
                event = events.get_nowait()
                if isinstance(event, tuple):
                    pronunciation_events.append(event)
                else:
                    last_event = event
        except queue.Empty:
            pass

        for event in pronunciation_events:
            if event[0] == "check_pronunciation":
                start_pronunciation_check(event[1])
            elif event[0] == "pronunciation_result":
                show_pronunciation_result(*event[1:])
            
        if last_event == "translate_sel":
            handle_translate_sel()
        elif last_event == "translate_ocr":
            handle_translate_ocr()
        elif last_event == "speak_sel":
            handle_speak_sel()
        elif last_event == "show_history":
            history_win.toggle(popup.root)
            
        # Check again after 80ms
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
                popup.show("Could not get selected text.", x, y)
                return
            popup.show_loading(x, y)
            popup.root.update_idletasks()
            
            sentences = split_sentences(text)
            results = mgr.translate(sentences)
            
            if not results:
                final_text = "(no results)"
            elif len(results) == 1:
                final_text = results[0]
            else:
                final_text = "\n---\n".join(f"[{i+1}] {r}" for i, r in enumerate(results))
                
            popup.show(final_text, x, y, source_text=text)
            if results:
                history_db.save(text, final_text, method="selection")
                if tts_cfg.get("auto_speak"):
                    tts_engine.speak_async(results[0], "vi")
        except Exception as e:
            popup.show(f"Translation error: {e}", x, y)
        finally:
            is_translating = False

    def handle_translate_ocr():
        nonlocal is_translating
        if is_translating:
            return
            
        is_translating = True
        x, y = None, None  # Initialize x, y to prevent UnboundLocalError
        try:
            # Call OCR
            text, x_new, y_new = ocr.capture_and_ocr(ocr_engine_inst)
            if text is None:
                # User cancelled or error
                return
            x, y = x_new, y_new
            if not text:
                popup.show("No text recognized in selected area.", x, y)
                return
                
            popup.show_loading(x, y)
            popup.root.update_idletasks()
            
            sentences = split_sentences(text)
            results = mgr.translate(sentences)
            
            if not results:
                final_text = "(no results)"
            elif len(results) == 1:
                final_text = results[0]
            else:
                final_text = "\n---\n".join(f"[{i+1}] {r}" for i, r in enumerate(results))
                
            popup.show(final_text, x, y, source_text=text)
            if results:
                history_db.save(text, final_text, method="ocr")
                if tts_cfg.get("auto_speak"):
                    tts_engine.speak_async(results[0], "vi")
            
        except Exception as e:
            popup.show(f"OCR/Translation error: {e}", x, y)
        finally:
            is_translating = False

    def handle_speak_sel():
        try:
            text = get_selected_text()
            x, y = mouse.position
            if not text:
                popup.show("Could not get selected text to pronounce.", x, y)
                return
            # Pronounce the selected text immediately
            tts_engine.speak_async(text)
            # Show the text in popup
            popup.show(f"\U0001f50a {text}", x, y, source_text=text)
        except Exception as e:
            popup.show(f"Pronounce error: {e}", x, y)

    def start_pronunciation_check(reference):
        nonlocal is_checking_pronunciation
        if is_checking_pronunciation:
            return

        is_checking_pronunciation = True
        x, y = mouse.position
        popup.show("Listening... Please read the English text aloud.", x, y, source_text=reference)
        duration = pronunciation_cfg.get("record_seconds", 5)
        sample_rate = pronunciation_cfg.get("sample_rate", 16000)

        def check_in_background():
            try:
                audio = record_audio(duration, sample_rate)
                result = pronunciation_evaluator.evaluate(reference, audio, sample_rate)
                events.put(("pronunciation_result", reference, result, None, x, y))
            except Exception as e:
                events.put(("pronunciation_result", reference, None, str(e), x, y))

        threading.Thread(target=check_in_background, daemon=True).start()

    def show_pronunciation_result(reference, result, error, x, y):
        nonlocal is_checking_pronunciation
        is_checking_pronunciation = False
        if error:
            popup.show(f"Pronunciation error: {error}", x, y, source_text=reference)
            return
        popup.show(
            reference,
            x,
            y,
            source_text=reference,
            pronunciation_result=result,
        )

    popup.root.after(80, process_queue)
    try:
        popup.run_mainloop()
    finally:
        tts_engine.stop()
        listener_sel.stop()
        listener_ocr.stop()
        if listener_speak:
            listener_speak.stop()


if __name__ == "__main__":
    main()