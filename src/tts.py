"""
Local pretrained Piper text-to-speech engine.
"""

import os
import io
import re
import threading
import queue
import time
import wave
from pathlib import Path

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
pygame = None
try:
    import pygame

    pygame_available = True
except ImportError:
    pygame_available = False

piper_voice = None
try:
    from piper import PiperVoice as piper_voice

    piper_available = True
except ImportError:
    piper_available = False


def is_vietnamese(text: str) -> bool:
    """Heuristic check if text is likely Vietnamese."""
    vi_chars = r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]"
    return bool(re.search(vi_chars, text, re.IGNORECASE))


class TTSEngine:
    def __init__(self, config=None):
        cfg = config or {}
        self.enabled = cfg.get("enabled", True)
        self.mode = cfg.get("engine", "piper")
        self.rate = cfg.get("rate", "+0%")
        self.volume = cfg.get("volume", "+0%")
        project_root = Path(__file__).resolve().parents[1]
        default_voices = {
            "en": str(project_root / "models" / "tts" / "en_US-lessac-medium.onnx"),
            "vi": str(project_root / "models" / "tts" / "vi_VN-vais1000-medium.onnx"),
        }
        self.voices = {**default_voices, **cfg.get("voices", {})}

        self.mixer_initialized = False
        if pygame_available and pygame is not None:
            try:
                # Initialize pygame mixer with standard audio settings
                pygame.mixer.init()
                self.mixer_initialized = True
            except Exception as e:
                print(f"[TTS] Warning: Could not initialize pygame mixer: {e}")

        self.piper_voices = {}
        self.voice_lock = threading.Lock()

        # Task queue and worker thread
        self.task_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.worker_thread = threading.Thread(target=self.worker_loop, daemon=True)
        self.worker_thread.start()

    def get_voice(self, lang):
        if not piper_available or piper_voice is None:
            print("[TTS] Piper is not installed. Run: pip install piper-tts")
            return None
        if lang in self.piper_voices:
            return self.piper_voices[lang]

        model_path = Path(self.voices.get(lang, self.voices["en"])).expanduser()
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[1] / model_path
        if not model_path.is_file():
            print(f"[TTS] Piper model not found: {model_path}")
            return None

        with self.voice_lock:
            if lang not in self.piper_voices:
                try:
                    self.piper_voices[lang] = piper_voice.load(str(model_path))
                except Exception as e:
                    print(f"[TTS] Could not load Piper model {model_path}: {e}")
                    return None
        return self.piper_voices[lang]

    def stop(self):
        """Immediately stop any current playback and clear pending queue."""
        self.stop_event.set()
        # Drain the queue
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
                self.task_queue.task_done()
            except queue.Empty:
                break

        if pygame_available and pygame is not None and self.mixer_initialized:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

    def speak_async(self, text, lang=None):
        """Queue text for playback in background thread."""
        if not self.enabled or not text or not text.strip():
            return

        text = text.strip()
        # Clean markdown / numbered prefixes like [1], [2] if present
        text = re.sub(r"^\[\d+\]\s*", "", text)

        # Stop previous sound if any before queueing new speech
        self.stop()
        self.stop_event.clear()

        # Determine language if not supplied
        if not lang:
            lang = "vi" if is_vietnamese(text) else "en"
        else:
            lang = "vi" if ("vi" in lang.lower() or "vie" in lang.lower()) else "en"

        self.task_queue.put((text, lang))

    def worker_loop(self):
        """Background thread executing speech requests one by one."""
        while True:
            try:
                text, lang = self.task_queue.get()
                if self.stop_event.is_set():
                    self.task_queue.task_done()
                    continue

                self.speak_internal(text, lang)
                self.task_queue.task_done()
            except Exception as e:
                print(f"[TTS] Worker error: {e}")

    def speak_internal(self, text, lang):
        self.speak_piper(text, lang)

    def speak_piper(self, text, lang):
        """Synthesize with a cached local Piper voice and play the WAV stream."""
        if self.stop_event.is_set():
            return

        voice = self.get_voice(lang)
        if voice is None or pygame is None or not self.mixer_initialized:
            return

        audio_fp = io.BytesIO()
        try:
            with wave.open(audio_fp, "wb") as wav_file:
                voice.synthesize(text, wav_file)
            audio_fp.seek(0)
        except Exception as e:
            print(f"[TTS] Piper synthesis failed: {e}")
            return

        if self.stop_event.is_set():
            return

        try:
            pygame.mixer.music.load(audio_fp, "wav")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self.stop_event.is_set():
                    pygame.mixer.music.stop()
                    return
                time.sleep(0.05)
        except Exception as e:
            print(f"[TTS] Pygame playback error: {e}")
