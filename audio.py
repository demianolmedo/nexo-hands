"""Audio: welcome/goodbye playback + double-clap detection via microphone."""
import os
import subprocess
import threading
import time
import numpy as np
import sounddevice as sd  # noqa: F401 — used for synth beeps below


ASSETS = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
WELCOME = os.path.join(ASSETS, "welcome.wav")
ACTIVATION_SONG = os.path.join(ASSETS, "activation.mp3")
SEEYA = os.path.join(ASSETS, "seeya.wav")


def _play_async(path: str, duration_sec: float | None = None, volume: float = 0.5):
    """Play an audio file in background via afplay (macOS native)."""
    if not os.path.exists(path):
        print(f"[audio] file missing: {path}")
        return None
    args = ["afplay"]
    if volume is not None:
        args.extend(["-v", f"{volume:.2f}"])
    if duration_sec is not None:
        args.extend(["-t", str(duration_sec)])
    args.append(path)
    try:
        return subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception as e:
        print(f"[audio] afplay failed: {e}")
        return None


_current_procs: list[subprocess.Popen] = []


def _stop_all():
    for p in _current_procs:
        try:
            p.terminate()
        except Exception:
            pass
    _current_procs.clear()
    # Extra: kill any stray afplay we may have spawned in rapid fire
    try:
        subprocess.run(["pkill", "-f", "afplay"], capture_output=True, timeout=1)
    except Exception:
        pass


def play_welcome_sequence():
    """welcome.wav → then 35s of activation.mp3, both in background without blocking."""
    def run():
        _stop_all()
        p1 = _play_async(WELCOME, volume=0.45)
        if p1:
            _current_procs.append(p1)
            p1.wait()
        p2 = _play_async(ACTIVATION_SONG, duration_sec=20, volume=0.45)
        if p2:
            _current_procs.append(p2)
    threading.Thread(target=run, daemon=True).start()


def play_seeya():
    def run():
        _stop_all()
        p = _play_async(SEEYA, volume=0.45)
        if p:
            _current_procs.append(p)
    threading.Thread(target=run, daemon=True).start()


# ------------- SYNTH BEEPS (zero-dependency state change chimes) -------------

def _synth_tone(start_hz: float, end_hz: float, duration_sec: float = 0.25, volume: float = 0.35):
    """Play a short sine-wave tone that glides from start_hz to end_hz. Non-blocking."""
    def run():
        try:
            sr = 22050
            n = int(sr * duration_sec)
            t = np.linspace(0, duration_sec, n, endpoint=False)
            # Linear frequency glide
            freq = np.linspace(start_hz, end_hz, n)
            phase = np.cumsum(2 * np.pi * freq / sr)
            wave = np.sin(phase).astype(np.float32) * volume
            # Quick fade-in/out to avoid clicks
            fade = min(int(sr * 0.02), n // 4)
            if fade > 0:
                env = np.ones(n, dtype=np.float32)
                env[:fade] = np.linspace(0, 1, fade)
                env[-fade:] = np.linspace(1, 0, fade)
                wave *= env
            sd.play(wave, samplerate=sr, blocking=False)
        except Exception as e:
            print(f"[audio] synth_tone failed: {e}")
    threading.Thread(target=run, daemon=True).start()


def beep_gestures_on():
    """Ascending chirp — gestures turning on."""
    _synth_tone(420, 720, duration_sec=0.22)


def beep_gestures_off():
    """Descending chirp — gestures turning off."""
    _synth_tone(720, 380, duration_sec=0.22)


def beep_voice_on_overlay():
    """Quick double-click — voice turning on while gestures already on."""
    _synth_tone(600, 900, duration_sec=0.12)
    threading.Timer(0.16, lambda: _synth_tone(800, 1100, duration_sec=0.12)).start()


def beep_gestures_resumed():
    """Soft single note — gestures reactivate after voice-off."""
    _synth_tone(520, 680, duration_sec=0.18)


# ------------- CLAP DETECTION -------------

class ClapDetector:
    """Consumes audio chunks from the shared `mic` broker.
    Two close-together loud peaks within (min_gap, max_gap) fire on_double_clap."""

    def __init__(self, threshold=0.15, min_gap=0.18, max_gap=0.8, on_double_clap=None):
        self.threshold = threshold
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.on_double_clap = on_double_clap
        self.last_peak_ts = 0.0
        self.peak_count = 0
        self._running = False

    def _on_chunk(self, chunk: np.ndarray):
        if not self._running:
            return
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        now = time.time()
        if rms > self.threshold:
            gap_since_last = now - self.last_peak_ts
            if gap_since_last < self.min_gap:
                return
            self.last_peak_ts = now
            self.peak_count += 1
            if self.peak_count >= 2 and gap_since_last <= self.max_gap:
                self.peak_count = 0
                if self.on_double_clap:
                    try:
                        self.on_double_clap()
                    except Exception as e:
                        print(f"[clap] handler error: {e}")
        else:
            if now - self.last_peak_ts > self.max_gap:
                self.peak_count = 0

    def start(self):
        if self._running:
            return
        self._running = True
        import mic as _mic
        if not _mic.is_started():
            _mic.start()
        _mic.subscribe(self._on_chunk)

    def stop(self):
        self._running = False
        import mic as _mic
        _mic.unsubscribe(self._on_chunk)
