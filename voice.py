"""Voice commands for nexo-hands.

Architecture:
  1. faster-whisper tiny (local) detects wake words: "despierta nexo" / "descansa nexo".
     Audio does not leave the machine until wake word is recognized.
  2. Once awake, audio is streamed live to Gemini Live Preview via WebSocket
     (asyncio inside a dedicated thread so the main OpenCV loop keeps running).
     Gemini returns function calls (tool use) that we dispatch.
  3. Simple commands are first matched against a local regex pre-filter — zero
     latency, zero API cost, zero audio leaving the machine for those commands.
  4. If the Live WebSocket cannot connect, falls back to Gemini 2.0 Flash text
     via standard generate_content.

Secrets: the Gemini API key is read from .env via python-dotenv — NEVER hardcoded.
"""
import os
import re
import time
import asyncio
import threading
import subprocess
from pathlib import Path

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv

from google import genai
from google.genai import types as genai_types


# ----------------- CONFIG (no secrets in code) -----------------

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

SAMPLE_RATE = 16000
# Gemini 3.1 Flash — newest generation available (verified via API Apr 2026)
LIVE_MODEL = "gemini-3.1-flash-live-preview"          # WebSocket Live API, bidi audio
FALLBACK_MODEL = "gemini-3.1-flash-lite-preview"      # standard generate_content path

# Wake words — simplified to single verbs that Whisper transcribes reliably.
# Whisper tiny often mis-hears proper names in Spanish audio, so we dropped the
# name requirement and use just a verb stem. Match is case-insensitive.
OPEN_VERBS = ("descansa", "descanse", "descansan", "descanza", "descanzan", "descanso")
CLOSE_VERBS = ("duerme", "duerma", "duermen", "duermo", "durmi", "durmiendo")


# ----------------- FUNCTION DECLARATIONS -----------------

TOOLS = [{
    "function_declarations": [
        {"name": "open_url", "description": "Abre una URL en el navegador por defecto",
         "parameters": {"type": "object", "properties": {"url": {"type": "string"}}, "required": ["url"]}},
        {"name": "open_app", "description": "Abre una aplicación de macOS por su nombre (ej Spotify, Chrome, Notion)",
         "parameters": {"type": "object", "properties": {"app_name": {"type": "string"}}, "required": ["app_name"]}},
        {"name": "type_text", "description": "Escribe texto en la ventana activa. NO presiona Enter.",
         "parameters": {"type": "object", "properties": {"text": {"type": "string"}}, "required": ["text"]}},
        {"name": "type_in_window",
         "description": "Escribe texto en una ventana específica identificada por fragmento de su título. NO presiona Enter por defecto.",
         "parameters": {"type": "object", "properties": {
             "window_name_contains": {"type": "string"},
             "text": {"type": "string"},
             "press_enter": {"type": "boolean"}
         }, "required": ["window_name_contains", "text"]}},
        {"name": "send_text", "description": "Presiona Enter en la ventana activa", "parameters": {"type": "object", "properties": {}}},
        {"name": "search_google", "description": "Nueva pestaña de Chrome con búsqueda en Google",
         "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        {"name": "play_pause", "description": "Toggle play/pause media key", "parameters": {"type": "object", "properties": {}}},
        {"name": "next_track", "description": "Siguiente canción", "parameters": {"type": "object", "properties": {}}},
        {"name": "previous_track", "description": "Canción anterior", "parameters": {"type": "object", "properties": {}}},
        {"name": "copy_clipboard", "description": "Cmd+C", "parameters": {"type": "object", "properties": {}}},
        {"name": "paste_clipboard", "description": "Cmd+V", "parameters": {"type": "object", "properties": {}}},
        {"name": "query_web",
         "description": "Responde por audio una pregunta del usuario. Úsala cuando el usuario pide info (dime, dame, explica, qué, cuál).",
         "parameters": {"type": "object", "properties": {"question": {"type": "string"}}, "required": ["question"]}},
    ]
}]

BLOCKED_SHELL_PATTERNS = [
    "rm ", "rm -", "sudo", "chmod", "chown", "dd ", "mkfs", "> /", "| sh", "curl | sh"
]


# ----------------- ACTION EXECUTORS -----------------

def _osa(script: str) -> str:
    r = subprocess.run(["osascript", "-e", script], capture_output=True, text=True, timeout=5)
    return r.stdout.strip()


def _type_string(text: str):
    safe = text.replace("\\", "\\\\").replace('"', '\\"')
    _osa(f'tell application "System Events" to keystroke "{safe}"')


def _find_window_pid_by_title(title_contains: str) -> int | None:
    try:
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


def _speak_mac(text: str, voice: str = "Monica"):
    try:
        subprocess.Popen(["say", "-v", voice, text[:500]],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
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
    from pynput.keyboard import Key, Controller
    kb = Controller()
    kb.press(Key.media_play_pause); kb.release(Key.media_play_pause)
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
    try:
        client = genai.Client(api_key=GEMINI_API_KEY)
        response = client.models.generate_content(
            model=FALLBACK_MODEL,
            contents=f"Responde en español de forma breve (máximo 3 frases, 50 palabras). Pregunta: {question}",
        )
        answer = ""
        for cand in (response.candidates or []):
            for part in (cand.content.parts or []):
                if getattr(part, "text", None):
                    answer = (answer + " " + part.text).strip()
        if answer:
            _speak_mac(answer)
            return f"QUERY: {answer[:60]}..."
        return "QUERY (no response)"
    except Exception as e:
        exec_search_google(question)
        return f"QUERY→search fallback: {str(e)[:40]}"


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


# ----------------- REGEX PRE-FILTER -----------------

_KNOWN_APPS = {
    "chrome": "Google Chrome", "google chrome": "Google Chrome",
    "safari": "Safari", "spotify": "Spotify", "notion": "Notion",
    "terminal": "Terminal", "iterm": "iTerm", "finder": "Finder",
    "music": "Music", "mail": "Mail", "slack": "Slack",
    "telegram": "Telegram", "whatsapp": "WhatsApp", "zoom": "zoom.us",
    "vscode": "Visual Studio Code", "code": "Visual Studio Code",
    "xcode": "Xcode", "preview": "Preview", "calculator": "Calculator",
    "calendar": "Calendar", "notes": "Notes", "reminders": "Reminders",
}


def _regex_prefilter(text: str):
    t = text.strip().lower()
    if re.search(r"\b(pausa|pause|para|stop)\b", t) and "nexo" not in t[:15]:
        return exec_play_pause(), "PAUSE"
    if re.search(r"\b(reproduce|play|reanuda|continúa|continua)\b", t):
        return exec_play_pause(), "PLAY"
    if re.search(r"\b(siguiente|next|próxima|proxima)\b", t):
        return exec_next_track(), "NEXT"
    if re.search(r"\b(anterior|previous|previa)\b", t):
        return exec_previous_track(), "PREV"
    if re.fullmatch(r"\s*(copia|copiar|copy)\s*\.?\s*", t):
        return exec_copy(), "COPY"
    if re.fullmatch(r"\s*(pega|pegar|paste)\s*\.?\s*", t):
        return exec_paste(), "PASTE"
    m = re.search(r"\b(abre|abrir|open|lanza|launch)\s+(.{2,40})", t)
    if m:
        target = m.group(2).strip(".,?! ")
        if target.startswith("http") or re.match(r"^[\w.-]+\.(com|org|net|io|app|dev|ai|co|es|ar|bo|ve|mx)(/.*)?$", target):
            return exec_open_url(target), f"URL {target[:30]}"
        for key, app_name in _KNOWN_APPS.items():
            if key in target.lower():
                return exec_open_app(app_name), f"APP {app_name}"
    m = re.search(r"\b(busca|buscar|search)\s+(.+)", t)
    if m:
        query = m.group(2).strip(".,?! ")
        return exec_search_google(query), f"SEARCH {query[:30]}"
    return None


def _try_regex(text: str):
    try:
        return _regex_prefilter(text)
    except Exception as e:
        print(f"[voice] regex prefilter error: {e}")
        return None


# ----------------- AUDIO BUFFER -----------------

class AudioBuffer:
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


# ----------------- GEMINI LIVE SESSION -----------------

class GeminiLiveSession:
    """Runs an asyncio event loop in its own thread. Exposes thread-safe
    methods to send audio chunks and receive function-call results via callback.

    If the Live API fails to connect, degrades to text-based fallback via
    standard generate_content in send_text_command().
    """

    def __init__(self, api_key: str, on_action, on_transcript):
        self.api_key = api_key
        self.on_action = on_action
        self.on_transcript = on_transcript
        self._thread: threading.Thread | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._session = None
        self._audio_queue: asyncio.Queue | None = None
        self._running = False
        self._connected = False
        self._fallback_mode = False

    def start(self):
        if self._running:
            return
        self._running = True
        self._thread = threading.Thread(target=self._run_loop, daemon=True)
        self._thread.start()

    def _run_loop(self):
        try:
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            self._loop.run_until_complete(self._main())
        except Exception as e:
            print(f"[voice-live] event loop crashed: {e}")
            self._fallback_mode = True

    async def _main(self):
        self._audio_queue = asyncio.Queue(maxsize=100)
        client = genai.Client(
            api_key=self.api_key,
            http_options={"api_version": "v1alpha"},
        )

        config = {
            # native-audio model requires AUDIO output modality
            "response_modalities": ["AUDIO"],
            "tools": TOOLS,
            "system_instruction": (
                "Eres Ivan, asistente de voz en macOS. Interpretás voz del usuario "
                "y ejecutás funciones. Cuando el usuario pida algo que matche alguna "
                "función disponible, llamála sin narrar de más. Para queries de "
                "información (dime, dame, explica), usá query_web. Las funciones "
                "type_text y type_in_window NO presionan Enter por defecto; solo si "
                "el usuario dice 'y envía' / 'y ejecuta' / 'y manda' usá press_enter=True. "
                "Respondé en español, con voz breve y amable."
            ),
        }

        try:
            async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
                self._session = session
                self._connected = True
                print(f"[voice-live] connected to {LIVE_MODEL}")
                send_task = asyncio.create_task(self._send_audio_loop())
                recv_task = asyncio.create_task(self._recv_loop())
                await asyncio.gather(send_task, recv_task)
        except Exception as e:
            print(f"[voice-live] connect failed ({e}); fallback to text mode")
            self._fallback_mode = True
            self._connected = False
            while self._running:
                try:
                    await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
                except asyncio.TimeoutError:
                    continue

    async def _send_audio_loop(self):
        while self._running:
            try:
                chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
                if chunk is None:
                    continue
                await self._session.send_realtime_input(
                    audio={"data": chunk.tobytes(), "mime_type": "audio/pcm;rate=16000"}
                )
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"[voice-live] send error: {e}")
                await asyncio.sleep(0.5)

    async def _recv_loop(self):
        try:
            async for response in self._session.receive():
                if hasattr(response, "text") and response.text:
                    self.on_transcript(response.text)
                tc = getattr(response, "tool_call", None)
                if tc and getattr(tc, "function_calls", None):
                    for fc in tc.function_calls:
                        await self._handle_function_call(fc)
                sc = getattr(response, "server_content", None)
                if sc:
                    mt = getattr(sc, "model_turn", None)
                    if mt and getattr(mt, "parts", None):
                        for p in mt.parts:
                            fc = getattr(p, "function_call", None)
                            if fc and fc.name:
                                await self._handle_function_call(fc)
        except Exception as e:
            print(f"[voice-live] recv loop error: {e}")

    async def _handle_function_call(self, fc):
        name = fc.name
        args = dict(fc.args or {})
        fn = FN_DISPATCH.get(name)
        if not fn:
            self.on_action(f"UNKNOWN FN {name}")
            return
        try:
            result = fn(**args)
            self.on_action(result)
            try:
                await self._session.send_tool_response(
                    function_responses=[{
                        "name": name,
                        "response": {"result": result},
                        "id": getattr(fc, "id", None),
                    }]
                )
            except Exception:
                pass
        except Exception as e:
            self.on_action(f"ERROR {name}: {e}")

    def push_audio(self, audio: np.ndarray):
        if not self._connected or self._audio_queue is None or self._loop is None:
            return
        pcm16 = (np.clip(audio, -1.0, 1.0) * 32767).astype(np.int16)
        try:
            asyncio.run_coroutine_threadsafe(self._audio_queue.put(pcm16), self._loop)
        except Exception:
            pass

    # Defensive rate limit so transient Gemini errors don't hammer the HUD
    _last_err_flash_at: float = 0.0

    def send_text_command(self, text: str):
        """Fallback: non-Live Gemini with tools. Skip if model unavailable."""
        if not self.api_key:
            return
        try:
            client = genai.Client(api_key=self.api_key)
            response = client.models.generate_content(
                model=FALLBACK_MODEL,
                contents=text,
                config=genai_types.GenerateContentConfig(
                    system_instruction=(
                        "Eres Ivan. Identifica qué función ejecutar según la voz del usuario. "
                        "Solo press_enter=True si el usuario dice 'y envía' o 'y ejecuta'. "
                        "Si no hay función clara, no llames nada."
                    ),
                    tools=TOOLS,
                ),
            )
            for cand in (response.candidates or []):
                for part in (cand.content.parts or []):
                    fc = getattr(part, "function_call", None)
                    if fc and fc.name:
                        fn = FN_DISPATCH.get(fc.name)
                        if fn:
                            try:
                                result = fn(**dict(fc.args or {}))
                                self.on_action(result + " (fallback)")
                                return
                            except Exception as e:
                                self.on_action(f"ERROR {fc.name}: {e}")
                                return
        except Exception as e:
            import time as _t
            now = _t.time()
            if now - self._last_err_flash_at > 5.0:
                self._last_err_flash_at = now
                self.on_action(f"GEMINI ERR: {str(e)[:80]}")

    def is_fallback(self) -> bool:
        return self._fallback_mode

    def stop(self):
        self._running = False
        if self._loop and self._audio_queue is not None:
            try:
                asyncio.run_coroutine_threadsafe(self._audio_queue.put(None), self._loop)
            except Exception:
                pass


# ----------------- MAIN VOICE SYSTEM -----------------

class VoiceSystem:
    def __init__(self, on_state_change=None, on_transcript=None, on_action=None):
        self.model = None
        self.audio_buf = AudioBuffer(seconds=3.0)
        self.on_state_change = on_state_change or (lambda s: None)
        self.on_transcript = on_transcript or (lambda t: None)
        self.on_action = on_action or (lambda a: None)

        self._awake = False
        self._last_activity = 0.0
        self._last_command_at = 0.0
        self._running = False
        self._scan_thread: threading.Thread | None = None
        self._live: GeminiLiveSession | None = None

        if not GEMINI_API_KEY:
            print("[voice] WARNING: GEMINI_API_KEY not set in .env — Gemini disabled")

    def _ensure_model(self):
        if self.model is None:
            self.model = WhisperModel("tiny", device="cpu", compute_type="int8")

    def _transcribe(self, audio: np.ndarray, language: str | None = None) -> str:
        self._ensure_model()
        segments, _ = self.model.transcribe(
            audio, language=language, beam_size=1, vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 300},
        )
        return " ".join(s.text.strip() for s in segments).strip().lower()

    def _on_chunk(self, chunk: np.ndarray):
        """Called by the shared mic broker for each audio chunk."""
        self.audio_buf.push(chunk)
        if self._awake and self._live is not None and not self._live.is_fallback():
            self._live.push_audio(chunk)

    def _scan_loop(self):
        """Background thread: transcribe buffer with Whisper to detect wake words.
        In fallback mode, also dispatch commands via non-Live Gemini."""
        last_check = 0.0
        while self._running:
            now = time.time()
            if now - last_check < 1.2:
                time.sleep(0.1)
                continue
            last_check = now

            audio = self.audio_buf.snapshot()
            if np.max(np.abs(audio)) < 0.01:
                continue

            try:
                text = self._transcribe(audio, language="es")
            except Exception as e:
                print(f"[voice] transcribe error: {e}")
                continue

            if not text:
                continue

            self.on_transcript(text[:60])
            print(f"[voice] heard: {text!r}")

            has_open_verb = any(v in text for v in OPEN_VERBS)
            has_close_verb = any(v in text for v in CLOSE_VERBS)

            if not self._awake and has_open_verb:
                self._activate()
            elif self._awake and has_close_verb:
                self._deactivate()
            elif self._awake and now - self._last_command_at > 3.0:
                self._last_command_at = now
                self._last_activity = now
                lower = text.lower()
                blocked = False
                for pat in BLOCKED_SHELL_PATTERNS:
                    if pat in lower:
                        self.on_action(f"BLOCKED: '{pat}'")
                        blocked = True
                        break
                if blocked:
                    continue
                regex_hit = _try_regex(text)
                if regex_hit is not None:
                    self.on_action(regex_hit[1] + " (local)")
                    continue
                if self._live and self._live.is_fallback():
                    self._live.send_text_command(text)

    def _activate(self):
        if self._awake:
            return
        self._awake = True
        self._last_activity = time.time()
        self._last_command_at = time.time()
        if GEMINI_API_KEY and self._live is None:
            self._live = GeminiLiveSession(
                api_key=GEMINI_API_KEY,
                on_action=self.on_action,
                on_transcript=self.on_transcript,
            )
            self._live.start()
        self.on_state_change("awake")
        print("[voice] WAKE")

    def _deactivate(self):
        if not self._awake:
            return
        self._awake = False
        if self._live is not None:
            self._live.stop()
            self._live = None
        self.on_state_change("sleep")
        print("[voice] SLEEP")

    def check_idle_sleep(self, idle_sec: float = 30.0):
        if self._awake and time.time() - self._last_activity > idle_sec:
            self._deactivate()

    def is_awake(self) -> bool:
        return self._awake

    def start(self):
        if self._running:
            return
        self._ensure_model()
        self._running = True
        import mic as _mic
        if not _mic.is_started():
            _mic.start()
        _mic.subscribe(self._on_chunk)
        self._scan_thread = threading.Thread(target=self._scan_loop, daemon=True)
        self._scan_thread.start()
        print("[voice] system started (wake: 'descansa' → ON / 'duerme' → OFF)")

    def stop(self):
        self._running = False
        import mic as _mic
        _mic.unsubscribe(self._on_chunk)
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass
