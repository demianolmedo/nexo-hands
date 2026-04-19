"""Singleton microphone broker.

macOS Core Audio reports err=-50 when two `sd.InputStream` instances try to
capture the same device simultaneously. This module owns a single InputStream
and fans out the audio chunks to any number of subscribers (clap detector,
Whisper wake-word scanner, Gemini Live uploader, ...).

Sample rate is fixed at 16 kHz mono — high enough for Whisper (which resamples
to 16 kHz anyway), low enough to keep CPU use tiny, and perfectly fine for RMS
peak detection used by the clap detector.
"""
import threading
import numpy as np
import sounddevice as sd


SAMPLE_RATE = 16000
BLOCKSIZE = 1024

_stream: sd.InputStream | None = None
_subscribers: list = []
_lock = threading.Lock()
_started = False


def _callback(indata, frames, t, status):
    # Copy once, fan out to all subscribers
    chunk = indata[:, 0].astype(np.float32)
    with _lock:
        subs = tuple(_subscribers)
    for cb in subs:
        try:
            cb(chunk)
        except Exception as e:
            print(f"[mic] subscriber error: {e}")


def start():
    """Idempotent — safe to call multiple times."""
    global _stream, _started
    if _started:
        return
    _stream = sd.InputStream(
        callback=_callback,
        channels=1,
        samplerate=SAMPLE_RATE,
        blocksize=BLOCKSIZE,
    )
    _stream.start()
    _started = True
    print(f"[mic] single shared InputStream started @ {SAMPLE_RATE} Hz")


def stop():
    global _stream, _started, _subscribers
    if _stream:
        try:
            _stream.stop()
            _stream.close()
        except Exception:
            pass
        _stream = None
    _started = False
    with _lock:
        _subscribers = []


def subscribe(callback):
    """Register a callback(chunk: np.ndarray) that receives audio in real time."""
    with _lock:
        if callback not in _subscribers:
            _subscribers.append(callback)


def unsubscribe(callback):
    with _lock:
        if callback in _subscribers:
            _subscribers.remove(callback)


def is_started() -> bool:
    return _started
