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


# ------------- SYNTH BEEPS (pre-generated WAV files → afplay, no sd conflict) -------------

def _write_chirp_wav(path: str, start_hz: float, end_hz: float, duration_sec: float, volume: float = 0.6):
    """Generate a sine-wave chirp WAV file at `path` (no-op if file already exists)."""
    if os.path.exists(path):
        return
    sr = 22050
    n = int(sr * duration_sec)
    freq = np.linspace(start_hz, end_hz, n)
    phase = np.cumsum(2 * np.pi * freq / sr)
    wave = np.sin(phase).astype(np.float32) * volume
    fade = min(int(sr * 0.02), n // 4)
    if fade > 0:
        env = np.ones(n, dtype=np.float32)
        env[:fade] = np.linspace(0, 1, fade)
        env[-fade:] = np.linspace(1, 0, fade)
        wave *= env
    pcm16 = (wave * 32767).astype(np.int16)
    try:
        import wave as _wave
        with _wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)
            wf.setframerate(sr)
            wf.writeframes(pcm16.tobytes())
    except Exception as e:
        print(f"[audio] chirp write failed: {e}")


_CHIRP_DIR = os.path.join(os.path.dirname(__file__), "assets", "_chirps")
os.makedirs(_CHIRP_DIR, exist_ok=True)
_CHIRP_GESTURES_OFF = os.path.join(_CHIRP_DIR, "gestures_off.wav")
_CHIRP_VOICE_OVERLAY = os.path.join(_CHIRP_DIR, "voice_overlay.wav")
_CHIRP_RESUMED = os.path.join(_CHIRP_DIR, "resumed.wav")
_write_chirp_wav(_CHIRP_GESTURES_OFF, 720, 380, 0.25)
_write_chirp_wav(_CHIRP_VOICE_OVERLAY, 700, 1100, 0.18)
_write_chirp_wav(_CHIRP_RESUMED, 520, 680, 0.20)


def _synth_tone(start_hz: float = None, end_hz: float = None,
                duration_sec: float = 0.25, volume: float = 0.5):
    """Legacy no-op — kept so old callers don't error. Use one of the beep_* fns."""
    pass


def beep_gestures_on():
    """Gestures turning on — short welcome chime (uses welcome.wav for natural feel)."""
    def run():
        p = _play_async(WELCOME, volume=0.55)
        if p:
            _current_procs.append(p)
    threading.Thread(target=run, daemon=True).start()


def beep_gestures_off():
    """Gestures off — descending chirp via afplay."""
    def run():
        p = _play_async(_CHIRP_GESTURES_OFF, volume=0.55)
        if p:
            _current_procs.append(p)
    threading.Thread(target=run, daemon=True).start()


def beep_voice_on_overlay():
    """Voice on while gestures already on — rising chirp via afplay."""
    def run():
        p = _play_async(_CHIRP_VOICE_OVERLAY, volume=0.5)
        if p:
            _current_procs.append(p)
    threading.Thread(target=run, daemon=True).start()


def beep_gestures_resumed():
    """Voice off, gestures resuming — soft ascending note via afplay."""
    def run():
        p = _play_async(_CHIRP_RESUMED, volume=0.45)
        if p:
            _current_procs.append(p)
    threading.Thread(target=run, daemon=True).start()


# ------------- CLAP DETECTION -------------

class ClapDetector:
    """Consumes audio chunks from the shared `mic` broker.
    Two close-together sharp peaks within (min_gap, max_gap) fire on_double_clap.

    Sharp vs voice: a clap is a very short attack (<30 ms) with high peak/rms
    ratio; speech is sustained. We gate on both (a) a high RMS threshold,
    (b) a high peak/rms ratio, and (c) a post-startup silence grace so the
    stream warmup doesn't look like a clap.
    """

    def __init__(self, threshold=0.28, min_gap=0.18, max_gap=0.8,
                 peak_to_rms_min=3.0, startup_grace_sec=2.0, on_double_clap=None):
        self.threshold = threshold
        self.min_gap = min_gap
        self.max_gap = max_gap
        self.peak_to_rms_min = peak_to_rms_min
        self.startup_grace_sec = startup_grace_sec
        self.on_double_clap = on_double_clap
        self.last_peak_ts = 0.0
        self.peak_count = 0
        self._running = False
        self._start_ts = 0.0

    def _on_chunk(self, chunk: np.ndarray):
        if not self._running:
            return
        now = time.time()
        # Ignore the first couple seconds after startup — mic warmup produces clicks
        if now - self._start_ts < self.startup_grace_sec:
            return
        rms = float(np.sqrt(np.mean(chunk ** 2)))
        peak = float(np.max(np.abs(chunk)))
        # Require both a loud peak AND that peak be much louder than the rms
        # (claps are transients; speech is not).
        is_clap_like = (
            peak > self.threshold
            and rms > 0
            and (peak / rms) >= self.peak_to_rms_min
        )
        if is_clap_like:
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
        self._start_ts = time.time()
        import mic as _mic
        if not _mic.is_started():
            _mic.start()
        _mic.subscribe(self._on_chunk)

    def stop(self):
        self._running = False
        import mic as _mic
        _mic.unsubscribe(self._on_chunk)
