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
from typing import Optional, Dict, Any

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
pygame: Any = None
try:
    import pygame

    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

PiperVoice: Any = None
try:
    from piper import PiperVoice

    _PIPER_AVAILABLE = True
except ImportError:
    _PIPER_AVAILABLE = False


def is_vietnamese(text: str) -> bool:
    """Heuristic check if text is likely Vietnamese."""
    vi_chars = r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]"
    return bool(re.search(vi_chars, text, re.IGNORECASE))


class TTSEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
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

        self._mixer_initialized = False
        if _PYGAME_AVAILABLE:
            try:
                # Initialize pygame mixer with standard audio settings
                pygame.mixer.init()
                self._mixer_initialized = True
            except Exception as e:
                print(f"[TTS] Warning: Could not initialize pygame mixer: {e}")

        self._piper_voices = {}
        self._voice_lock = threading.Lock()

        # Task queue and worker thread
        self._task_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _get_voice(self, lang: str):
        if not _PIPER_AVAILABLE:
            print("[TTS] Piper is not installed. Run: pip install piper-tts")
            return None
        if lang in self._piper_voices:
            return self._piper_voices[lang]

        model_path = Path(self.voices.get(lang, self.voices["en"])).expanduser()
        if not model_path.is_absolute():
            model_path = Path(__file__).resolve().parents[1] / model_path
        if not model_path.is_file():
            print(f"[TTS] Piper model not found: {model_path}")
            return None

        with self._voice_lock:
            if lang not in self._piper_voices:
                try:
                    self._piper_voices[lang] = PiperVoice.load(str(model_path))
                except Exception as e:
                    print(f"[TTS] Could not load Piper model {model_path}: {e}")
                    return None
        return self._piper_voices[lang]

    def stop(self):
        """Immediately stop any current playback and clear pending queue."""
        self._stop_event.set()
        # Drain the queue
        while not self._task_queue.empty():
            try:
                self._task_queue.get_nowait()
                self._task_queue.task_done()
            except queue.Empty:
                break

        if _PYGAME_AVAILABLE and self._mixer_initialized:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

    def speak_async(self, text: str, lang: Optional[str] = None):
        """Queue text for playback in background thread."""
        if not self.enabled or not text or not text.strip():
            return

        text = text.strip()
        # Clean markdown / numbered prefixes like [1], [2] if present
        text = re.sub(r"^\[\d+\]\s*", "", text)

        # Stop previous sound if any before queueing new speech
        self.stop()
        self._stop_event.clear()

        # Determine language if not supplied
        if not lang:
            lang = "vi" if is_vietnamese(text) else "en"
        else:
            lang = "vi" if ("vi" in lang.lower() or "vie" in lang.lower()) else "en"

        self._task_queue.put((text, lang))

    def _worker_loop(self):
        """Background thread executing speech requests one by one."""
        while True:
            try:
                text, lang = self._task_queue.get()
                if self._stop_event.is_set():
                    self._task_queue.task_done()
                    continue

                self._speak_internal(text, lang)
                self._task_queue.task_done()
            except Exception as e:
                print(f"[TTS] Worker error: {e}")

    def _speak_internal(self, text: str, lang: str):
        self._speak_piper(text, lang)

    def _speak_piper(self, text: str, lang: str):
        """Synthesize with a cached local Piper voice and play the WAV stream."""
        if self._stop_event.is_set():
            return

        voice = self._get_voice(lang)
        if voice is None or not self._mixer_initialized:
            return

        audio_fp = io.BytesIO()
        try:
            with wave.open(audio_fp, "wb") as wav_file:
                voice.synthesize(text, wav_file)
            audio_fp.seek(0)
        except Exception as e:
            print(f"[TTS] Piper synthesis failed: {e}")
            return

        if self._stop_event.is_set():
            return

        try:
            pygame.mixer.music.load(audio_fp, "wav")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self._stop_event.is_set():
                    pygame.mixer.music.stop()
                    return
                time.sleep(0.05)
        except Exception as e:
            print(f"[TTS] Pygame playback error: {e}")
