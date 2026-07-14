"""
Register and listen for global hotkeys using pynput.

When the user presses a hotkey, it just pushes an "event" into the queue.
The translation + popup logic is handled by main.py on the main thread (as required by tkinter).
"""
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


class HotkeyListener:
    def __init__(self, hotkey_str: str, on_trigger):
        """on_trigger: a parameterless function, called when the hotkey is pressed."""
        self.combo = parse_hotkey(hotkey_str)
        self.on_trigger = on_trigger
        self._listener = None

    def start(self):
        self._listener = keyboard.GlobalHotKeys({self.combo: self.on_trigger})
        self._listener.start()  # runs on a separate thread

    def stop(self):
        if self._listener is not None:
            self._listener.stop()