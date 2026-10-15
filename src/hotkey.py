"""
Register and listen for global hotkeys using pynput.

When the user presses a hotkey, it just pushes an "event" into the queue.
The translation + popup logic is handled by main.py on the main thread (as required by tkinter).
"""
import re
import time

from pynput import keyboard


def parse_hotkey(hotkey_str: str) -> str:
    """ 'ctrl+shift+c' -> pynput format '<ctrl>+<shift>+c' """
    mods = {"ctrl": "<ctrl>", "shift": "<shift>", "alt": "<alt>",
            "cmd": "<cmd>", "win": "<cmd>"}
    keys = []
    for k in hotkey_str.lower().split("+"):
        k = k.strip()
        keys.append(mods.get(k, k))
    return "+".join(keys)


def parse_double_tap(hotkey_str: str):
    """Return the key for a repeated single-character hotkey such as 'a+a'."""
    parts = [part.strip().lower() for part in hotkey_str.split("+")]
    if len(parts) == 2 and parts[0] == parts[1] and re.fullmatch(r"[a-z0-9]", parts[0]):
        return parts[0]
    return None


class HotkeyListener:
    def __init__(self, hotkey_str: str, on_trigger, double_tap_interval: float = 0.4):
        """on_trigger: a parameterless function, called when the hotkey is pressed."""
        self.double_tap_key = parse_double_tap(hotkey_str)
        self.combo = None if self.double_tap_key else parse_hotkey(hotkey_str)
        self.on_trigger = on_trigger
        self.double_tap_interval = double_tap_interval
        self._listener = None
        self._key_is_down = False
        self._last_tap_at = None

    def start(self):
        if self.double_tap_key:
            self._listener = keyboard.Listener(
                on_press=self._on_press,
                on_release=self._on_release,
            )
        else:
            self._listener = keyboard.GlobalHotKeys({self.combo: self.on_trigger})
        self._listener.start()

    def _on_press(self, key):
        if getattr(key, "char", None) != self.double_tap_key or self._key_is_down:
            return

        self._key_is_down = True
        now = time.monotonic()
        if (
            self._last_tap_at is not None
            and now - self._last_tap_at <= self.double_tap_interval
        ):
            self._last_tap_at = None
            self.on_trigger()
        else:
            self._last_tap_at = now

    def _on_release(self, key):
        if getattr(key, "char", None) == self.double_tap_key:
            self._key_is_down = False

    def stop(self):
        if self._listener is not None:
            self._listener.stop()