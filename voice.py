"""Voice commands for nexo-hands: "despierta nexo" wake word via Whisper local,
then Gemini Flash interprets commands and executes functions.

Flow:
  1. Mic stream → Whisper tiny scans 3s windows for "despierta nexo" / "descansa nexo"
  2. On wake → record 6s → Whisper transcribe → Gemini with function calling
  3. Gemini returns function call → dispatch to actions

Functions defined (Gemini picks which to call):
  open_url, open_app, type_text, type_in_window, send_text,
  search_google, play_pause, next_track, previous_track,
  copy_clipboard, paste_clipboard, query_web
"""
import os
import time
import threading
import subprocess
import queue
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel

from google import genai
from google.genai import types as genai_types


GEMINI_API_KEY = "REDACTED_API_KEY"
SAMPLE_RATE = 16000
WAKE_WORD_OPEN = "despierta nexo"
WAKE_WORD_CLOSE = "descansa nexo"

# Gemini function declarations — Gemini chooses which one to call
TOOLS = [{
    "function_declarations": [
        {
            "name": "open_url",
            "description": "Abre una URL en el navegador por defecto",
            "parameters": {
                "type": "object",
                "properties": {"url": {"type": "string", "description": "URL completa a abrir"}},
                "required": ["url"]
            }
        },
        {
            "name": "open_app",
            "description": "Abre una aplicación de macOS por su nombre",
            "parameters": {
                "type": "object",
                "properties": {"app_name": {"type": "string", "description": "Nombre de la app, por ejemplo 'Spotify', 'Chrome', 'Notion'"}},
                "required": ["app_name"]
            }
        },
        {
            "name": "type_text",
            "description": "Escribe texto en la ventana que está actualmente activa (frontmost). NO presiona Enter.",
            "parameters": {
                "type": "object",
                "properties": {"text": {"type": "string"}},
                "required": ["text"]
            }
        },
        {
            "name": "type_in_window",
            "description": "Escribe texto en una ventana específica identificada por parte de su título. NO presiona Enter por defecto.",
            "parameters": {
                "type": "object",
                "properties": {
                    "window_name_contains": {"type": "string", "description": "Fragmento del título, por ejemplo 'claude-demian', 'Chrome', 'Terminal'"},
                    "text": {"type": "string"},
                    "press_enter": {"type": "boolean", "description": "Si debe presionar Enter después. Por default False."}
                },
                "required": ["window_name_contains", "text"]
            }
        },
        {
            "name": "send_text",
            "description": "Presiona Enter en la ventana activa (útil después de type_text para enviar)",
            "parameters": {"type": "object", "properties": {}}
        },
        {
            "name": "search_google",
            "description": "Abre una nueva pestaña de Chrome y busca el query en Google",
            "parameters": {
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"]
            }
        },
        {"name": "play_pause", "description": "Toggle play/pause en la app de audio activa", "parameters": {"type": "object", "properties": {}}},
        {"name": "next_track", "description": "Siguiente canción (media key)", "parameters": {"type": "object", "properties": {}}},
        {"name": "previous_track", "description": "Canción anterior (media key)", "parameters": {"type": "object", "properties": {}}},
        {"name": "copy_clipboard", "description": "Copiar selección actual (Cmd+C)", "parameters": {"type": "object", "properties": {}}},
        {"name": "paste_clipboard", "description": "Pegar (Cmd+V)", "parameters": {"type": "object", "properties": {}}},
        {
            "name": "query_web",
            "description": "Responde una pregunta buscando en web. Usa esto cuando el usuario pide información (ej. 'dime las noticias de hoy', 'qué clima hace'). La respuesta se lee por audio.",
            "parameters": {
                "type": "object",
                "properties": {"question": {"type": "string"}},
                "required": ["question"]
            }
        },
    ]
}]

BLOCKED_SHELL_PATTERNS = [
    "rm ", "rm -", "sudo", "chmod", "chown", "dd ", "mkfs", "> /", "| sh", "curl | sh"
]


# ----------------- AUDIO UTILS -----------------

class AudioBuffer:
    """Rolling buffer of last N seconds of audio for wake word scanning."""
    def __init__(self, seconds: float = 3.0, sr: int = SAMPLE_RATE):
        self.max_samples = int(seconds * sr)
        self.buf = np.zeros(self.max_samples, dtype=np.float32)
        self.sr = sr
        self._lock = threading.Lock()

    def push(self, chunk: np.ndarray):
        with self._lock:
            n = len(chunk)
            if n >= self.max_samples:
                self.buf = chunk[-self.max_samples:].copy()
            else:
                self.buf = np.concatenate([self.buf[n:], chunk])

    def snapshot(self) -> np.ndarray:
        with self._lock:
            return self.buf.copy()


# ----------------- ACTION EXECUTORS -----------------

def _osa(script: str) -> str:
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5)
    return r.stdout.strip()


def _type_string(text: str):
    """Type a string using AppleScript keystroke. Escapes quotes."""
    safe = text.replace("\\", "\\\\").replace('"', '\\"')
    _osa(f'tell application "System Events" to keystroke "{safe}"')


def _find_window_pid_by_title(title_contains: str) -> int | None:
    """Return PID of the app owning the topmost window matching title fragment."""
    try:
        import windows  # our local module
        from Quartz import CGWindowListCopyWindowInfo, kCGWindowListOptionOnScreenOnly, kCGNullWindowID
        wins = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
        needle = title_contains.lower()
        for w in wins:
            if w.get("kCGWindowLayer", 0) != 0:
                continue
            name = str(w.get("kCGWindowName", "") or "")
            owner = str(w.get("kCGWindowOwnerName", "") or "")
            if needle in name.lower() or needle in owner.lower():
                return int(w.get("kCGWindowOwnerPID", 0))
    except Exception:
        pass
    return None


def _focus_pid(pid: int):
    try:
        from AppKit import NSRunningApplication
        app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
        if app:
            app.activateWithOptions_(2)
            time.sleep(0.2)
    except Exception:
        pass


def exec_open_url(url: str) -> str:
    if not url.startswith(("http://", "https://")):
        url = "https://" + url
    subprocess.run(["open", url], check=False)
    return f"OPEN URL {url[:50]}"


def exec_open_app(app_name: str) -> str:
    subprocess.run(["open", "-a", app_name], check=False)
    return f"OPEN APP {app_name}"


def exec_type_text(text: str) -> str:
    _type_string(text)
    return f"TYPE ({len(text)} chars)"


def exec_type_in_window(window_name_contains: str, text: str, press_enter: bool = False) -> str:
    pid = _find_window_pid_by_title(window_name_contains)
    if pid is not None:
        _focus_pid(pid)
    _type_string(text)
    if press_enter:
        time.sleep(0.15)
        _osa('tell application "System Events" to key code 36')
    return f"TYPE in {window_name_contains} ({len(text)} chars){' + ENTER' if press_enter else ''}"


def exec_send_text() -> str:
    _osa('tell application "System Events" to key code 36')
    return "ENTER"


def exec_search_google(query: str) -> str:
    import urllib.parse
    url = "https://www.google.com/search?q=" + urllib.parse.quote(query)
    subprocess.run(["open", "-a", "Google Chrome", url], check=False)
    return f"SEARCH {query[:40]}"


def exec_play_pause() -> str:
    try:
        from pynput.keyboard import Key, Controller
        kb = Controller()
        kb.press(Key.media_play_pause); kb.release(Key.media_play_pause)
    except Exception:
        pass
    return "PLAY/PAUSE"


def exec_next_track() -> str:
    from pynput.keyboard import Key, Controller
    kb = Controller()
    kb.press(Key.media_next); kb.release(Key.media_next)
    return "NEXT"


def exec_previous_track() -> str:
    from pynput.keyboard import Key, Controller
    kb = Controller()
    kb.press(Key.media_previous); kb.release(Key.media_previous)
    return "PREV"


def exec_copy() -> str:
    _osa('tell application "System Events" to keystroke "c" using command down')
    return "COPY"


def exec_paste() -> str:
    _osa('tell application "System Events" to keystroke "v" using command down')
    return "PASTE"


def exec_query_web(question: str) -> str:
    # For MVP we just delegate to a google search — audio-answer is left for v2 (requires TTS)
    return exec_search_google(question)


FN_DISPATCH = {
    "open_url": exec_open_url,
    "open_app": exec_open_app,
    "type_text": exec_type_text,
    "type_in_window": exec_type_in_window,
    "send_text": exec_send_text,
    "search_google": exec_search_google,
    "play_pause": exec_play_pause,
    "next_track": exec_next_track,
    "previous_track": exec_previous_track,
    "copy_clipboard": exec_copy,
    "paste_clipboard": exec_paste,
    "query_web": exec_query_web,
}


# ----------------- VOICE SYSTEM -----------------

class VoiceSystem:
    def __init__(self, on_state_change=None, on_transcript=None, on_action=None):
        self.model = None  # lazy load
        self.audio_buf = AudioBuffer(seconds=3.0)
        self.on_state_change = on_state_change or (lambda s: None)
        self.on_transcript = on_transcript or (lambda t: None)
        self.on_action = on_action or (lambda a: None)

        self._awake = False
        self._last_activity = 0.0
        self._running = False
        self._stream: sd.InputStream | None = None
        self._last_wake_check = 0.0
        self._last_command_at = 0.0
        self._scan_thread: threading.Thread | None = None
        self._scan_queue: queue.Queue = queue.Queue()
        self._gemini_client: genai.Client | None = None

    def _ensure_model(self):
        if self.model is None:
            self.model = WhisperModel("tiny", device="cpu", compute_type="int8")

    def _ensure_gemini(self):
        if self._gemini_client is None:
            self._gemini_client = genai.Client(api_key=GEMINI_API_KEY)

    def _transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        self._ensure_model()
        segments, _ = self.model.transcribe(
            audio, language=language, beam_size=1, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        return " ".join(s.text.strip() for s in segments).strip().lower()

    def _audio_callback(self, indata, frames, t, status):
        chunk = indata[:, 0].astype(np.float32)
        self.audio_buf.push(chunk)

    def _scan_loop(self):
        """Background thread: periodically transcribe the buffer looking for wake words."""
        last_check = 0.0
        while self._running:
            now = time.time()
            if now - last_check < 1.2:  # check every ~1.2 seconds
                time.sleep(0.1)
                continue
            last_check = now

            audio = self.audio_buf.snapshot()
            if np.max(np.abs(audio)) < 0.01:
                continue  # silent, skip

            try:
                text = self._transcribe(audio, language="es")
            except Exception as e:
                print(f"[voice] transcribe error: {e}")
                continue

            if not text:
                continue

            print(f"[voice] buffer: '{text}'")
            self.on_transcript(text)

            if not self._awake and any(w in text for w in ["despierta nexo", "despierta, nexo", "nexo despierta"]):
                self._activate()
            elif self._awake and any(w in text for w in ["descansa nexo", "descansa, nexo", "nexo descansa"]):
                self._deactivate()
            elif self._awake and now - self._last_command_at > 3.0:
                # Treat it as a command
                self._handle_command(text)

    def _activate(self):
        if self._awake:
            return
        self._awake = True
        self._last_activity = time.time()
        self._last_command_at = time.time()
        self.on_state_change("awake")
        print("[voice] WAKE")

    def _deactivate(self):
        if not self._awake:
            return
        self._awake = False
        self.on_state_change("sleep")
        print("[voice] SLEEP")

    def _handle_command(self, text: str):
        """Send text to Gemini for function calling; execute whatever it returns."""
        self._last_command_at = time.time()
        self._last_activity = time.time()

        # Security: pre-check for blocked shell patterns
        lower = text.lower()
        for pat in BLOCKED_SHELL_PATTERNS:
            if pat in lower:
                self.on_action(f"BLOCKED: '{pat}' no permitido")
                return

        self._ensure_gemini()

        try:
            system_prompt = (
                "Eres Nexo, un asistente de voz para macOS. Recibes texto transcrito del usuario. "
                "Tu trabajo es identificar qué función ejecutar. Si el usuario da un comando claro, "
                "llama a la función apropiada. Si no es claro, no llames ninguna función. "
                "Las funciones type_text y type_in_window NO presionan Enter por defecto. "
                "Solo presiona Enter (press_enter=true) si el usuario dice explícitamente "
                "'y envía', 'y ejecuta', 'y manda'. "
                "El usuario mezcla español e inglés; entiende ambos."
            )
            response = self._gemini_client.models.generate_content(
                model="gemini-2.0-flash-exp",
                contents=text,
                config=genai_types.GenerateContentConfig(
                    system_instruction=system_prompt,
                    tools=TOOLS,
                ),
            )

            # Find function call in response
            for cand in (response.candidates or []):
                for part in (cand.content.parts or []):
                    fc = getattr(part, "function_call", None)
                    if fc and fc.name:
                        fn = FN_DISPATCH.get(fc.name)
                        if fn:
                            try:
                                args = dict(fc.args or {})
                                result = fn(**args)
                                self.on_action(result)
                                return
                            except Exception as e:
                                self.on_action(f"ERROR {fc.name}: {e}")
                                return
            # No function call returned — fall through silently
        except Exception as e:
            self.on_action(f"GEMINI ERROR: {str(e)[:80]}")

    def check_idle_sleep(self, idle_sec: float = 30.0):
        """Called from main loop. Auto-sleeps if awake but no activity."""
        if self._awake and time.time() - self._last_activity > idle_sec:
            self._deactivate()

    def is_awake(self) -> bool:
        return self._awake

    def start(self):
        if self._running:
            return
        self._ensure_model()
        self._running = True
        self._stream = sd.InputStream(
            callback=self._audio_callback, channels=1,
            samplerate=SAMPLE_RATE, blocksize=4096,
        )
        self._stream.start()
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()
        print("[voice] system started")

    def stop(self):
        self._running = False
        if self._stream:
            try:
                self._stream.stop(); self._stream.close()
            except Exception:
                pass
            self._stream = None
