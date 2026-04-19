"""Media & system control. Uses pynput for HID media keys (works with YouTube, Spotify, Music)."""
import subprocess
from pynput.keyboard import Key, Controller

_kb = Controller()


def _tap(key):
    _kb.press(key)
    _kb.release(key)


def _osa(script: str) -> str:
    try:
        r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=2)
        return r.stdout.strip()
    except Exception:
        return ""


def play_pause():
    """Universal media play/pause — works in YouTube, Spotify, Music, Chrome video."""
    _tap(Key.media_play_pause)


def next_track():
    _tap(Key.media_next)


def previous_track():
    _tap(Key.media_previous)


def mute_toggle():
    _osa('set volume output muted (not (output muted of (get volume settings)))')


def volume_up(step: int = 5):
    current = _osa('output volume of (get volume settings)')
    try:
        v = int(current)
    except (ValueError, TypeError):
        v = 50
    new = max(0, min(100, v + step))
    _osa(f'set volume output volume {new}')


def volume_down(step: int = 5):
    current = _osa('output volume of (get volume settings)')
    try:
        v = int(current)
    except (ValueError, TypeError):
        v = 50
    new = max(0, min(100, v - step))
    _osa(f'set volume output volume {new}')


def switch_next_app():
    """Cmd+Tab → next app."""
    _osa('tell application "System Events" to keystroke tab using command down')


def desktop_left():
    """Ctrl+Left → previous desktop."""
    _osa('tell application "System Events" to key code 123 using control down')


def desktop_right():
    """Ctrl+Right → next desktop."""
    _osa('tell application "System Events" to key code 124 using control down')


def mission_control():
    """F3 → Mission Control."""
    _osa('tell application "System Events" to key code 160')
