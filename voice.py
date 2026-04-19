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

import json
import urllib.request
import urllib.error

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel
from dotenv import load_dotenv

# google-genai SDK is intentionally NOT imported at runtime: the preview build
# leaks multiprocessing semaphores on every call, which eventually trips SIGTRAP
# ("trace trap" + "resource_tracker: leaked semaphore"). We hit Gemini via plain
# REST calls in `_gemini_rest()` below — stdlib urllib, zero multiprocessing.


# ----------------- CONFIG (no secrets in code) -----------------

_HERE = Path(__file__).resolve().parent
load_dotenv(_HERE / ".env")
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

SAMPLE_RATE = 16000
# Gemini 3.1 Flash — newest generation available (verified via API Apr 2026)
LIVE_MODEL = "gemini-3.1-flash-live-preview"          # WebSocket Live API, bidi audio
FALLBACK_MODEL = "gemini-3.1-flash-lite-preview"      # standard generate_content path

# Wake words — single verbs, no name.
# OPEN  = "despierta"  (any inflection)
# CLOSE = "descansa"   (any inflection)
OPEN_VERBS = (
    "despierta", "despierte", "despierto", "despiertan",
    "espierta", "espierte", "despiertar",
)
CLOSE_VERBS = (
    "descansa", "descanse", "descansan", "descanza", "descanzan", "descanso",
    "descansar",
)


# ----------------- FUNCTION DECLARATIONS -----------------

# REST-style camelCase keys: `functionDeclarations`, `parametersJsonSchema`.
# The google-genai SDK auto-converted snake_case, but we're going REST now.
TOOLS = [{
    "functionDeclarations": [
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


# TTS provider: "gemini" (Gemini 3.1 Flash TTS Preview via REST, JARVIS-grade)
# or "mac" (macOS `say`, offline). Gemini fails back to mac on any error.
TTS_PROVIDER = os.environ.get("TTS_PROVIDER", "gemini").lower()
# Gemini voices (prebuilt): Orus, Charon, Fenrir = deep male.
# Zephyr, Puck, Algieba, Enceladus = other male options. Full list in docs.
TTS_MODEL_GEMINI = os.environ.get("TTS_MODEL_GEMINI", "gemini-3.1-flash-tts-preview")
TTS_VOICE_GEMINI = os.environ.get("TTS_VOICE_GEMINI", "Orus")
# macOS `say` fallback config
TTS_VOICE_MAC = os.environ.get("TTS_VOICE_MAC", "Reed (es_MX)")
TTS_RATE_MAC = int(os.environ.get("TTS_RATE_MAC", "185"))  # 175-200 conversacional
# Legacy alias so old callers keep working
TTS_VOICE = TTS_VOICE_MAC
TTS_RATE = TTS_RATE_MAC


def _kill_audio_output():
    """Kill any leftover say/afplay so the new utterance doesn't overlap."""
    for pat in (r"^say\b", "afplay"):
        try:
            subprocess.run(["pkill", "-f", pat], capture_output=True, timeout=1)
        except Exception:
            pass


def _speak_gemini(text: str) -> bool:
    """Gemini TTS via REST → WAV → afplay. Blocks until playback ends.
    Returns True on success, False on failure (caller falls back to `say`)."""
    import base64
    import tempfile
    import wave
    if not GEMINI_API_KEY:
        return False
    try:
        url = f"{_GEMINI_HOST}/models/{TTS_MODEL_GEMINI}:generateContent?key={GEMINI_API_KEY}"
        body = {
            "contents": [{"parts": [{"text": text[:800]}]}],
            "generationConfig": {
                "responseModalities": ["AUDIO"],
                "speechConfig": {
                    "voiceConfig": {
                        "prebuiltVoiceConfig": {"voiceName": TTS_VOICE_GEMINI}
                    }
                },
            },
        }
        req = urllib.request.Request(
            url,
            data=json.dumps(body).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=20) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        pcm = b""
        for cand in data.get("candidates", []):
            for part in cand.get("content", {}).get("parts", []):
                inline = part.get("inlineData") or part.get("inline_data")
                if inline and inline.get("data"):
                    pcm += base64.b64decode(inline["data"])
        if not pcm:
            return False
        _kill_audio_output()
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
            path = tf.name
        with wave.open(path, "wb") as wf:
            wf.setnchannels(1)
            wf.setsampwidth(2)       # 16-bit
            wf.setframerate(24000)   # Gemini TTS outputs 24 kHz PCM
            wf.writeframes(pcm)
        subprocess.run(
            ["afplay", path],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=30,
        )
        return True
    except Exception as e:
        print(f"[tts-gemini] fallback to say: {e}")
        return False


def _speak_mac(text: str, voice: str | None = None, rate: int | None = None):
    """macOS `say` fallback. Blocks until playback ends so the dispatch-thread
    STATE_SPEAKING flag stays set (prevents the scan loop from picking up
    our own voice via the mic and echoing another Gemini call)."""
    _kill_audio_output()
    voice = voice or TTS_VOICE_MAC
    rate = rate or TTS_RATE_MAC
    try:
        subprocess.run(
            ["say", "-v", voice, "-r", str(rate), text[:500]],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            timeout=30,
        )
    except Exception:
        pass


def _speak(text: str):
    """Main entry point: route to provider, fall back on failure."""
    if TTS_PROVIDER == "gemini" and _speak_gemini(text):
        return
    _speak_mac(text)


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


# ------------- GEMINI REST CLIENT (no SDK, no semaphore leaks) -------------

_GEMINI_HOST = "https://generativelanguage.googleapis.com/v1beta"


def _gemini_rest(model: str, user_text: str, system_instruction: str,
                 tools: list | None = None, thinking_high: bool = False,
                 timeout: float = 20.0) -> dict:
    """Plain HTTPS POST to the Gemini REST API. Returns parsed JSON.

    We use urllib from stdlib so there is zero multiprocessing involvement.
    The google-genai preview SDK leaks semaphores and has been the source of
    every SIGTRAP crash we've seen — this bypasses it completely."""
    if not GEMINI_API_KEY:
        raise RuntimeError("GEMINI_API_KEY not set")
    url = f"{_GEMINI_HOST}/models/{model}:generateContent?key={GEMINI_API_KEY}"
    body: dict = {
        "systemInstruction": {"parts": [{"text": system_instruction}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
    }
    if tools:
        body["tools"] = tools
    if thinking_high:
        body["generationConfig"] = {"thinkingConfig": {"thinkingLevel": "high"}}
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except urllib.error.HTTPError as e:
        detail = ""
        try:
            detail = e.read().decode("utf-8")[:200]
        except Exception:
            pass
        raise RuntimeError(f"gemini {e.code}: {detail}") from None


def _today_spanish() -> str:
    from datetime import datetime
    days_es = ["lunes", "martes", "miércoles", "jueves", "viernes", "sábado", "domingo"]
    months_es = ["enero", "febrero", "marzo", "abril", "mayo", "junio",
                 "julio", "agosto", "septiembre", "octubre", "noviembre", "diciembre"]
    now = datetime.now()
    return f"{days_es[now.weekday()]} {now.day} de {months_es[now.month - 1]} de {now.year}"


def exec_query_web(question: str) -> str:
    """Answer a conversational query via REST Gemini + macOS `say`. Uses
    thinking_level=high for better reasoning on nuanced questions."""
    try:
        system_inst = (
            "Sos Iván, asistente de voz en español. SIEMPRE respondés en "
            "ESPAÑOL NEUTRO LATINOAMERICANO, jamás en inglés ni en español "
            "de España (no uses 'vosotros', 'tío', 'ordenador'). Tono "
            "conversacional, cercano, natural — como si hablaras a un "
            f"amigo. Hoy es {_today_spanish()}; no inventes fechas. "
            "Máximo 3 frases, 60 palabras. Nada de markdown, asteriscos, "
            "listas ni emojis: tu respuesta se lee en voz alta."
        )
        data = _gemini_rest(
            model=FALLBACK_MODEL,
            user_text=question,
            system_instruction=system_inst,
            thinking_high=True,
        )
        answer = ""
        for cand in (data.get("candidates") or []):
            for part in (cand.get("content", {}).get("parts") or []):
                txt = part.get("text")
                if txt:
                    answer = (answer + " " + txt).strip()
        if not answer:
            return "QUERY (no response)"
        _speak(answer)
        return f"QUERY: {answer[:60]}..."
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


# ----------------- GEMINI LIVE SESSION (per official docs) -----------------

# Keywords that signal "the user wants a spoken answer" — only then do we
# play back the audio Gemini returns. For everything else (commands), we
# silently drain the PCM and only act on function_calls.
QUERY_KEYWORDS = (
    "dime", "dame", "hablame", "háblame", "hablemos", "cuéntame", "cuentame",
    "explícame", "explicame", "explica", "qué ", "que ", "cuál", "cual",
    "cómo ", "como ", "por qué", "por que", "cuánto", "cuanto",
)


def _is_query(text: str) -> bool:
    t = (text or "").lower()
    return any(kw in t for kw in QUERY_KEYWORDS)


class GeminiLiveSession:
    """Owns an asyncio event loop in its own thread. Re-implemented per the
    official Live API docs:
      https://ai.google.dev/gemini-api/docs/live-api/capabilities

    Key differences vs the earlier attempt that crashed with SIGTRAP:
      * Audio input wrapped in types.Blob (not a raw dict)
      * response_modalities=["AUDIO"] plus output_audio_transcription: {} so we
        receive the model's spoken reply as text too
      * Explicit handling of server_content.output_transcription,
        server_content.input_transcription, and server_content.turn_complete
      * Only play back PCM when the latest user command was a query; for
        command-style input we drain the bytes silently
      * Session refresh every ~14 minutes (Live sessions cap at 15 min)
      * After two consecutive connect failures, permanent fall-through to
        text-mode (send_text_command) so voice never goes dark
    """

    SESSION_LIFETIME_SEC = 14 * 60  # reconnect before the 15 min hard cap
    MAX_CONSECUTIVE_CONNECT_FAILURES = 2

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
        self._connect_failures = 0
        # Query-mode state (updated by main thread before it pushes audio)
        self._expect_audio_reply = False
        self._current_turn_audio: list[bytes] = []

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

    def set_expect_audio_reply(self, flag: bool):
        """Main thread tells us the current user turn is a query (wants voice)."""
        self._expect_audio_reply = flag

    async def _main(self):
        """Outer loop: repeatedly open a Live session until we're told to stop
        or fall into permanent text fallback."""
        self._audio_queue = asyncio.Queue(maxsize=200)

        while self._running:
            try:
                await self._run_session_once()
                # session_once returned cleanly → we just refreshed at 14 min
                # loop back and open a fresh one
                self._connect_failures = 0
            except Exception as e:
                self._connect_failures += 1
                print(f"[voice-live] session error ({self._connect_failures}/{self.MAX_CONSECUTIVE_CONNECT_FAILURES}): {e}")
                if self._connect_failures >= self.MAX_CONSECUTIVE_CONNECT_FAILURES:
                    print("[voice-live] permanent fallback to text mode")
                    self._fallback_mode = True
                    self._connected = False
                    # Drain any remaining queue to avoid backpressure on producers
                    while self._running:
                        try:
                            await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
                        except asyncio.TimeoutError:
                            continue
                    return
                await asyncio.sleep(1.0)

    async def _run_session_once(self):
        client = genai.Client(
            api_key=self.api_key,
            http_options={"api_version": "v1alpha"},
        )
        config = {
            "response_modalities": ["AUDIO"],
            "output_audio_transcription": {},
            "tools": TOOLS,
            "system_instruction": (
                "Eres Ivan, asistente de voz para macOS. Cuando el usuario pida "
                "ejecutar algo (abrir app, buscar, dictar, mover ventanas, play/pause), "
                "llamá la función correspondiente sin narrar ni dar explicación. "
                "Cuando el usuario te hable conversacionalmente (dime, explícame, "
                "cuéntame, hablemos, qué, cuál), respondé breve en español. "
                "type_text/type_in_window NO presionan Enter a menos que el usuario "
                "diga 'y envía' o 'y ejecuta'. Nunca ejecutes comandos destructivos."
            ),
        }

        session_start = time.time()
        try:
            async with client.aio.live.connect(model=LIVE_MODEL, config=config) as session:
                self._session = session
                self._connected = True
                self._connect_failures = 0
                print(f"[voice-live] connected to {LIVE_MODEL}")
                send_task = asyncio.create_task(self._send_audio_loop(session_start))
                recv_task = asyncio.create_task(self._recv_loop())
                # Wait for either task to finish (either by lifetime, error, or stop)
                done, pending = await asyncio.wait(
                    {send_task, recv_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for t in pending:
                    t.cancel()
                for t in done:
                    if t.exception():
                        raise t.exception()
        finally:
            self._connected = False
            self._session = None

    async def _send_audio_loop(self, session_start: float):
        """Send user audio to Gemini; exit to trigger a session refresh at ~14min."""
        while self._running:
            if time.time() - session_start > self.SESSION_LIFETIME_SEC:
                print("[voice-live] session lifetime reached, refreshing")
                return
            try:
                chunk = await asyncio.wait_for(self._audio_queue.get(), timeout=1.0)
                if chunk is None:
                    continue
                await self._session.send_realtime_input(
                    audio=genai_types.Blob(
                        data=chunk.tobytes(),
                        mime_type="audio/pcm;rate=16000",
                    ),
                )
            except asyncio.TimeoutError:
                continue
            except asyncio.CancelledError:
                raise
            except Exception as e:
                print(f"[voice-live] send error: {e}")
                await asyncio.sleep(0.5)

    async def _recv_loop(self):
        """Receive loop — drains audio, accumulates transcriptions, and routes
        tool calls. If the user asked a query, we play back the audio PCM when
        the turn finishes. Otherwise we discard the bytes."""
        try:
            async for response in self._session.receive():
                try:
                    sc = getattr(response, "server_content", None)
                    if sc is None:
                        # Top-level tool call (rare but possible)
                        tc = getattr(response, "tool_call", None)
                        if tc:
                            for fc in (getattr(tc, "function_calls", None) or []):
                                await self._safe_fc(fc)
                        continue

                    # User ASR transcript (what Gemini thinks we said)
                    it = getattr(sc, "input_transcription", None)
                    if it is not None:
                        t = getattr(it, "text", None)
                        if t:
                            self.on_transcript(str(t)[:120])

                    # Gemini's spoken reply, as text
                    ot = getattr(sc, "output_transcription", None)
                    if ot is not None:
                        t = getattr(ot, "text", None)
                        if t:
                            # Show it in the HUD action log
                            self.on_action(f"> {str(t)[:80]}")

                    mt = getattr(sc, "model_turn", None)
                    if mt is not None:
                        for p in (getattr(mt, "parts", None) or []):
                            inline = getattr(p, "inline_data", None)
                            if inline is not None:
                                data = getattr(inline, "data", None)
                                if data:
                                    # Accumulate PCM for the current turn; we'll
                                    # decide whether to play it at turn_complete.
                                    self._current_turn_audio.append(bytes(data))
                            fc = getattr(p, "function_call", None)
                            if fc is not None and getattr(fc, "name", None):
                                await self._safe_fc(fc)

                    if getattr(sc, "turn_complete", False):
                        # End of model's turn — maybe play back the audio
                        if self._expect_audio_reply and self._current_turn_audio:
                            pcm_bytes = b"".join(self._current_turn_audio)
                            self._play_pcm_bytes_async(pcm_bytes)
                        self._current_turn_audio.clear()
                        self._expect_audio_reply = False
                except Exception as inner:
                    print(f"[voice-live] part handler error: {inner}")
        except asyncio.CancelledError:
            raise
        except Exception as e:
            print(f"[voice-live] recv loop error: {e}")

    def _play_pcm_bytes_async(self, pcm_bytes: bytes):
        """Write PCM to a tmp WAV and afplay it. Runs in its own thread; non-blocking."""
        def run():
            try:
                import tempfile, wave, subprocess
                with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
                    path = tf.name
                with wave.open(path, "wb") as wf:
                    wf.setnchannels(1)
                    wf.setsampwidth(2)       # 16-bit
                    wf.setframerate(24000)   # Gemini Live audio out is 24 kHz
                    wf.writeframes(pcm_bytes)
                subprocess.Popen(["afplay", path], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            except Exception as e:
                print(f"[voice-live] tts playback failed: {e}")
        threading.Thread(target=run, daemon=True).start()

    async def _safe_fc(self, fc):
        try:
            await self._handle_function_call(fc)
        except Exception as e:
            print(f"[voice-live] fc error: {e}")

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

    # --- thread-safe audio push ---

    _push_buffer: list = []
    _push_samples = 0
    _PUSH_MIN_SAMPLES = 3200  # 200 ms @ 16 kHz

    def push_audio(self, audio: np.ndarray):
        if not self._connected or self._audio_queue is None or self._loop is None:
            return
        self._push_buffer.append(audio.astype(np.float32, copy=False))
        self._push_samples += len(audio)
        if self._push_samples < self._PUSH_MIN_SAMPLES:
            return
        combined = np.concatenate(self._push_buffer)
        self._push_buffer = []
        self._push_samples = 0
        pcm16 = (np.clip(combined, -1.0, 1.0) * 32767).astype(np.int16)
        try:
            asyncio.run_coroutine_threadsafe(self._audio_queue.put(pcm16), self._loop)
        except Exception:
            pass

    # Defensive rate limit so transient Gemini errors don't hammer the HUD
    _last_err_flash_at: float = 0.0

    def send_text_command(self, text: str):
        """Route `text` to Gemini via REST, dispatch the returned function call.
        REST (not SDK) avoids the semaphore-leak SIGTRAP."""
        if not self.api_key:
            return
        try:
            data = _gemini_rest(
                model=FALLBACK_MODEL,
                user_text=text,
                system_instruction=(
                    "Sos Iván, asistente de voz para macOS, hablás ESPAÑOL "
                    f"neutro latinoamericano. Hoy es {_today_spanish()}. "
                    "Identificá qué función ejecutar según lo que dijo el "
                    "usuario y llamala con los parámetros correctos. "
                    "Ejemplos: 'abre youtube' → open_url(url='youtube.com'). "
                    "'reproduce el video' → play_pause(). "
                    "'busca tal cosa' → search_google(query='tal cosa'). "
                    "Solo usá press_enter=True si el usuario dijo 'y envía' o "
                    "'y ejecuta'. Si no hay función clara, no llames nada "
                    "(no narres, no respondas en texto)."
                ),
                tools=TOOLS,
            )
            for cand in (data.get("candidates") or []):
                for part in (cand.get("content", {}).get("parts") or []):
                    # REST uses camelCase: functionCall with {name, args}
                    fc = part.get("functionCall")
                    if fc and fc.get("name"):
                        fn = FN_DISPATCH.get(fc["name"])
                        if fn:
                            try:
                                result = fn(**(fc.get("args") or {}))
                                self.on_action(result + " (fallback)")
                                return
                            except Exception as e:
                                self.on_action(f"ERROR {fc['name']}: {e}")
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
    # Processing states for the HUD indicator
    STATE_IDLE = "idle"          # not listening (asleep) OR listening silently
    STATE_LISTEN = "listen"      # actively transcribing user speech
    STATE_THINKING = "thinking"  # Gemini processing the command/query
    STATE_SPEAKING = "speaking"  # TTS audio is playing

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
        self._processing_state = self.STATE_IDLE
        self._state_lock = threading.Lock()
        # While the welcome/activation music plays, Whisper can hear the song
        # itself via the speaker → mic path and "transcribe" lyrics as commands.
        # We refuse to transcribe during this window.
        self._activation_blackout_until = 0.0

        if not GEMINI_API_KEY:
            print("[voice] WARNING: GEMINI_API_KEY not set in .env — Gemini disabled")

    def get_processing_state(self) -> str:
        with self._state_lock:
            return self._processing_state

    def _set_state(self, state: str):
        with self._state_lock:
            self._processing_state = state

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

            # If Gemini is thinking or we're speaking, don't transcribe — our
            # own TTS would be picked up by the mic and cause an echo loop
            # (double audio, runaway Gemini calls, eventually SIGTRAP).
            cur_state = self.get_processing_state()
            if cur_state in (self.STATE_THINKING, self.STATE_SPEAKING):
                self.audio_buf.buf[:] = 0.0  # drop whatever we captured of ourselves
                continue

            # Post-wake blackout while the activation ritual is audible — don't
            # let Whisper transcribe the welcome chime / music as if it were a
            # user command.
            if time.time() < self._activation_blackout_until:
                self.audio_buf.buf[:] = 0.0
                continue

            audio = self.audio_buf.snapshot()
            is_silent = np.max(np.abs(audio)) < 0.01
            # Only toggle IDLE/LISTEN based on voice activity; don't stomp on
            # THINKING/SPEAKING states driven by the dispatch thread.
            if cur_state in (self.STATE_IDLE, self.STATE_LISTEN):
                self._set_state(self.STATE_IDLE if is_silent else self.STATE_LISTEN)
            if is_silent:
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
            elif self._awake and now - self._last_command_at > 1.5:
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
                # Route to Gemini in a background thread so the scan_loop keeps
                # capturing audio and new commands aren't blocked while Gemini
                # thinks / TTS plays. Clear the audio buffer so the same
                # transcription doesn't get re-processed on the next scan.
                if self._live is not None:
                    is_query = _is_query(text)
                    self._live.set_expect_audio_reply(is_query)
                    # Clear recent audio so the same phrase doesn't fire twice
                    self.audio_buf.buf[:] = 0.0

                    def _dispatch(t=text, q=is_query):
                        self._set_state(
                            self.STATE_SPEAKING if q else self.STATE_THINKING
                        )
                        try:
                            if self._live and self._live.is_fallback():
                                self._live.send_text_command(t)
                        except Exception as e:
                            print(f"[voice] dispatch error: {e}")
                        finally:
                            # Drain any echo we captured of ourselves while
                            # Gemini/say were running before reopening the mic.
                            try:
                                self.audio_buf.buf[:] = 0.0
                            except Exception:
                                pass
                            self._last_command_at = time.time()
                            self._set_state(self.STATE_IDLE)
                    threading.Thread(target=_dispatch, daemon=True).start()

    def _activate(self):
        if self._awake:
            return
        self._awake = True
        self._last_activity = time.time()
        self._last_command_at = time.time()
        # Welcome (~2s) + activation music (15s) = ignore mic for ~17s so
        # Whisper doesn't transcribe the music.
        self._activation_blackout_until = time.time() + 17.0
        # NOTE: We bypass the Live WebSocket for now. The google-genai preview
        # SDK crashes with SIGTRAP when `gemini-3.1-flash-live-preview` returns
        # its first audio response on this stack, and that's been reproduced
        # across three implementation variants. We use the stable text-mode
        # path (Whisper wake → generate_content with tools → Gemini TTS for
        # query answers). Native Live streaming will come back when the SDK
        # stabilizes. Fallback mode also means send_text_command is used.
        if GEMINI_API_KEY:
            self._live = GeminiLiveSession(
                api_key=GEMINI_API_KEY,
                on_action=self.on_action,
                on_transcript=self.on_transcript,
            )
            self._live._fallback_mode = True
        self.on_state_change("awake")
        print("[voice] WAKE (text mode + Gemini TTS)")

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
        print("[voice] system started (wake: 'despierta' → ON / 'descansa' → OFF)")

    def stop(self):
        self._running = False
        import mic as _mic
        _mic.unsubscribe(self._on_chunk)
        if self._live:
            try:
                self._live.stop()
            except Exception:
                pass
