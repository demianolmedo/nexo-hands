"""Audio: welcome/goodbye playback + double-clap detection via microphone."""
import os
import subprocess
import threading
import time
import numpy as np
import sounddevice as sd


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


# ------------- CLAP DETECTION -------------

class ClapDetector:
    def __init__(self, threshold=0.15, min_gap=0.18, max_gap=0.8, on_double_clap=None):
        self.threshold = threshold
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.on_double_clap = on_double_clap
        self.last_peak_ts = 0.0
        self.last_reset_ts = 0.0
        self.peak_count = 0
        self._running = False
        self._stream = None

    def _callback(self, indata, frames, t, status):
        if status:
            return
        rms = float(np.sqrt(np.mean(indata ** 2)))
        now = time.time()
        # peak detection with refractory period (ignore very close events = echo)
        if rms > self.threshold:
            gap_since_last = now - self.last_peak_ts
            if gap_since_last < self.min_gap:
                return
            self.last_peak_ts = now
            self.peak_count += 1
            if self.peak_count >= 2 and gap_since_last <= self.max_gap:
                # double clap detected!
                self.peak_count = 0
                if self.on_double_clap:
                    try:
                        self.on_double_clap()
                    except Exception as e:
                        print(f"[clap] handler error: {e}")
        else:
            # decay peak counter if no recent clap
            if now - self.last_peak_ts > self.max_gap:
                self.peak_count = 0

    def start(self):
        if self._running:
            return
        self._running = True
        self._stream = sd.InputStream(
            callback=self._callback,
            channels=1,
            samplerate=22050,
            blocksize=1024,
        )
        self._stream.start()

    def stop(self):
        self._running = False
        if self._stream:
            try:
                self._stream.stop()
                self._stream.close()
            except Exception:
                pass
            self._stream = None
