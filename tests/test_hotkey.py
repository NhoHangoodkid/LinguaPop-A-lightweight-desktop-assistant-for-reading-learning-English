import unittest
from unittest.mock import patch

from src.hotkey import HotkeyListener, parse_double_tap, parse_hotkey


class HotkeyTests(unittest.TestCase):
    def test_parse_double_tap_binding(self):
        self.assertEqual(parse_double_tap("A+A"), "a")
        self.assertIsNone(parse_double_tap("ctrl+a"))
        self.assertIsNone(parse_double_tap("a+b"))

    def test_double_tap_triggers_once_after_key_release(self):
        triggered = []
        listener = HotkeyListener("a+a", lambda: triggered.append(True))
        key = type("Key", (), {"char": "a"})()

        with patch("src.hotkey.time.monotonic", side_effect=[1.0, 1.2]):
            listener._on_press(key)
            listener._on_release(key)
            listener._on_press(key)

        self.assertEqual(triggered, [True])

    def test_held_key_does_not_count_as_double_tap(self):
        triggered = []
        listener = HotkeyListener("a+a", lambda: triggered.append(True))
        key = type("Key", (), {"char": "a"})()

        with patch("src.hotkey.time.monotonic", return_value=1.0):
            listener._on_press(key)
            listener._on_press(key)

        self.assertEqual(triggered, [])

    def test_existing_modifier_hotkeys_are_preserved(self):
        listener = HotkeyListener("ctrl+shift+y", lambda: None)

        self.assertEqual(listener.combo, "<ctrl>+<shift>+y")
        self.assertEqual(parse_hotkey("ctrl+shift+y"), "<ctrl>+<shift>+y")


if __name__ == "__main__":
    unittest.main()
