"""
Text-to-Speech (TTS) Engine.
Supports Hybrid mode: Edge-TTS (high quality neural voice) with pyttsx3 (offline fallback).
Runs asynchronously in a background thread to prevent UI freezing.
"""
import os
import asyncio
import io
import re
import threading
import queue
import time
from typing import Optional, Dict, Any

os.environ["PYGAME_HIDE_SUPPORT_PROMPT"] = "1"
try:
    import pygame
    _PYGAME_AVAILABLE = True
except ImportError:
    _PYGAME_AVAILABLE = False

try:
    import edge_tts
    _EDGE_TTS_AVAILABLE = True
except ImportError:
    _EDGE_TTS_AVAILABLE = False

try:
    import pyttsx3
    _PYTTSX3_AVAILABLE = True
except ImportError:
    _PYTTSX3_AVAILABLE = False


def is_vietnamese(text: str) -> bool:
    """Heuristic check if text is likely Vietnamese."""
    vi_chars = r"[àáạảãâầấậẩẫăằắặẳẵèéẹẻẽêềếệểễìíịỉĩòóọỏõôồốộổỗơờớợởỡùúụủũưừứựửữỳýỵỷỹđĐ]"
    return bool(re.search(vi_chars, text, re.IGNORECASE))


class TTSEngine:
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        self.enabled = cfg.get("enabled", True)
        self.mode = cfg.get("engine", "hybrid")  # "hybrid", "edge-tts", "pyttsx3"
        self.rate = cfg.get("rate", "+0%")
        self.volume = cfg.get("volume", "+0%")
        self.voices = cfg.get("voices", {
            "en": "en-US-JennyNeural",
            "vi": "vi-VN-HoaiMyNeural"
        })

        self._mixer_initialized = False
        if _PYGAME_AVAILABLE:
            try:
                # Initialize pygame mixer with standard audio settings
                pygame.mixer.init()
                self._mixer_initialized = True
            except Exception as e:
                print(f"[TTS] Warning: Could not initialize pygame mixer: {e}")

        # pyttsx3 engine instance (initialized lazily)
        self._offline_engine = None
        self._offline_lock = threading.Lock()

        # Task queue and worker thread
        self._task_queue = queue.Queue()
        self._stop_event = threading.Event()
        self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._worker_thread.start()

    def _get_offline_engine(self):
        if self._offline_engine is None and _PYTTSX3_AVAILABLE:
            try:
                self._offline_engine = pyttsx3.init()
            except Exception as e:
                print(f"[TTS] Warning: Could not initialize pyttsx3: {e}")
        return self._offline_engine

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

        if self._offline_engine is not None:
            try:
                self._offline_engine.stop()
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
        voice = self.voices.get(lang, self.voices.get("en", "en-US-JennyNeural"))
        use_edge = (self.mode in ["hybrid", "edge-tts"]) and _EDGE_TTS_AVAILABLE and self._mixer_initialized

        spoken = False
        if use_edge:
            try:
                spoken = self._speak_edge_tts(text, voice)
            except Exception as e:
                print(f"[TTS] Edge-TTS failed ({e}), falling back to offline...")
                spoken = False

        if not spoken and (self.mode in ["hybrid", "pyttsx3"]):
            self._speak_offline(text, lang)

    def _speak_edge_tts(self, text: str, voice: str) -> bool:
        """Fetch audio stream from Edge-TTS and play using pygame."""
        if self._stop_event.is_set():
            return False

        async def _fetch():
            communicate = edge_tts.Communicate(text, voice, rate=self.rate, volume=self.volume)
            mp3_fp = io.BytesIO()
            async for chunk in communicate.stream():
                if self._stop_event.is_set():
                    return None
                if chunk["type"] == "audio":
                    mp3_fp.write(chunk["data"])
            mp3_fp.seek(0)
            return mp3_fp

        try:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            audio_fp = loop.run_until_complete(_fetch())
            loop.close()
        except Exception as e:
            return False

        if audio_fp is None or self._stop_event.is_set():
            return False

        try:
            pygame.mixer.music.load(audio_fp, "mp3")
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                if self._stop_event.is_set():
                    pygame.mixer.music.stop()
                    return False
                time.sleep(0.05)
            return True
        except Exception as e:
            print(f"[TTS] Pygame playback error: {e}")
            return False

    def _speak_offline(self, text: str, lang: str):
        """Fallback to pyttsx3 (SAPI5 on Windows)."""
        if self._stop_event.is_set():
            return

        with self._offline_lock:
            engine = self._get_offline_engine()
            if engine is None:
                return

            try:
                # Try to pick matching voice if available
                voices = engine.getProperty("voices")
                if voices:
                    target_voice = None
                    for v in voices:
                        name_lower = v.name.lower()
                        if lang == "vi" and ("vietnam" in name_lower or "vi-vn" in name_lower or "an" in name_lower):
                            target_voice = v.id
                            break
                        elif lang == "en" and ("english" in name_lower or "en-us" in name_lower or "zira" in name_lower or "david" in name_lower):
                            target_voice = v.id
                            break
                    if target_voice:
                        engine.setProperty("voice", target_voice)

                engine.say(text)
                engine.runAndWait()
            except Exception as e:
                print(f"[TTS] pyttsx3 error: {e}")
