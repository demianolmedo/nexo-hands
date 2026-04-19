"""Gesture-control HUD: status bar, side legends, action log, action flash, permission banner.
Minimalist, Iron Man / Minority Report aesthetic.
"""
import time
import cv2


COLOR_GREEN = (80, 255, 160)
COLOR_AMBER = (80, 200, 255)
COLOR_RED = (80, 80, 255)
COLOR_CYAN = (255, 220, 120)
COLOR_WHITE = (255, 255, 255)
COLOR_MUTED = (140, 140, 140)

GESTURE_COLORS = {
    "fist":   (80, 80, 255),
    "spread": (80, 255, 160),
    "ok":     (255, 200, 80),
    "peace":  (180, 120, 255),
    "rock":   (80, 255, 255),
    "three":  (120, 255, 120),
    "point":  (120, 180, 255),
    "pinch":  (200, 80, 255),
    "great":  (80, 255, 255),
    None:     (200, 200, 200),
}


_STATE_COLORS = {
    "idle":     (130, 130, 150),  # dim
    "listen":   (80, 255, 160),   # green — actively hearing user
    "thinking": (255, 200, 80),   # amber — waiting for Gemini
    "speaking": (255, 120, 200),  # pink/magenta — TTS playing
}
_STATE_LABEL = {
    "idle":     "",
    "listen":   "escuchando",
    "thinking": "pensando",
    "speaking": "hablando",
}


def draw_voice_indicator(frame, awake: bool, transcript: str = "", processing_state: str = "idle"):
    """Pulsing orb on the right side + state label + transcript, so the user
    always knows whether the system is listening, thinking or speaking."""
    h, w = frame.shape[:2]
    cx, cy = w - 40, h // 2

    if awake:
        color = _STATE_COLORS.get(processing_state, _STATE_COLORS["idle"])
        label = _STATE_LABEL.get(processing_state, "")

        # Pulse speed depends on state (fast during thinking/speaking)
        speed = 3.0 if processing_state in ("thinking", "speaking") else 2.0
        pulse = (time.time() * speed) % 1.0
        outer_r = int(22 + 14 * abs(0.5 - pulse) * 2)
        cv2.circle(frame, (cx, cy), outer_r, color, 2, cv2.LINE_AA)
        cv2.circle(frame, (cx, cy), 13, color, -1, cv2.LINE_AA)
        cv2.putText(frame, "NEXO", (cx - 22, cy + 36),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 1, cv2.LINE_AA)
        if label:
            ts = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.55, 1)[0]
            cv2.putText(frame, label, (cx - ts[0] // 2, cy + 55),
                        cv2.FONT_HERSHEY_DUPLEX, 0.55, color, 1, cv2.LINE_AA)
    else:
        cv2.circle(frame, (cx, cy), 8, (80, 80, 90), -1, cv2.LINE_AA)

    if transcript:
        # Movie-style caption: centered, larger, with translucent bar behind
        text = transcript[:70]
        font = cv2.FONT_HERSHEY_DUPLEX
        scale = 0.8
        thick = 2
        (tw, th), _ = cv2.getTextSize(text, font, scale, thick)
        pad_x = 24
        pad_y = 12
        y_baseline = h - 130
        x0 = (w - tw) // 2 - pad_x
        x1 = x0 + tw + 2 * pad_x
        y0 = y_baseline - th - pad_y
        y1 = y_baseline + pad_y
        overlay = frame.copy()
        cv2.rectangle(overlay, (x0, y0), (x1, y1), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)
        cv2.putText(frame, text, ((w - tw) // 2, y_baseline),
                    font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(frame, text, ((w - tw) // 2, y_baseline),
                    font, scale, (220, 230, 255), thick, cv2.LINE_AA)


def draw_status(frame, active: bool, attention_until: float,
                gesture_l: str | None, gesture_r: str | None,
                grabbed_app: str | None,
                layer_gestures: bool = False, layer_voice: bool = False):
    h, w = frame.shape[:2]
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (w, 54), (8, 8, 10), -1)
    cv2.addWeighted(overlay, 0.7, frame, 0.3, 0, frame)

    # Layer badges instead of one status blob
    gestures_color = COLOR_GREEN if layer_gestures else COLOR_MUTED
    voice_color = COLOR_CYAN if layer_voice else COLOR_MUTED

    cv2.circle(frame, (22, 27), 7, gestures_color, -1, cv2.LINE_AA)
    cv2.putText(frame, "GESTOS", (34, 33),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, gestures_color, 1, cv2.LINE_AA)
    cv2.circle(frame, (130, 27), 7, voice_color, -1, cv2.LINE_AA)
    cv2.putText(frame, "VOICE", (142, 33),
                cv2.FONT_HERSHEY_DUPLEX, 0.55, voice_color, 1, cv2.LINE_AA)

    # Show voice idle countdown if voice is on
    if layer_voice and attention_until < time.time() + 100:
        sec_left = max(0, int(attention_until - time.time()))
        cv2.putText(frame, f"{sec_left}s", (225, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_AMBER, 1, cv2.LINE_AA)

    # Paused gestures banner when voice is on while gestures_enabled=True
    if layer_voice and layer_gestures:
        cv2.putText(frame, "gestos pausados", (280, 33),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.48, (150, 150, 170), 1, cv2.LINE_AA)

    cv2.putText(frame, f"L {gesture_l or '-'}", (w - 260, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                GESTURE_COLORS.get(gesture_l, COLOR_MUTED), 1, cv2.LINE_AA)
    cv2.putText(frame, f"R {gesture_r or '-'}", (w - 130, 33),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                GESTURE_COLORS.get(gesture_r, COLOR_MUTED), 1, cv2.LINE_AA)

    if grabbed_app:
        banner = frame.copy()
        cv2.rectangle(banner, (0, 54), (w, 84), (30, 30, 36), -1)
        cv2.addWeighted(banner, 0.75, frame, 0.25, 0, frame)
        cv2.putText(frame, f"GRAB: {grabbed_app}", (20, 76),
                    cv2.FONT_HERSHEY_DUPLEX, 0.65, COLOR_CYAN, 2, cv2.LINE_AA)


def draw_side_legend(frame, side: str):
    """Full gesture cheatsheet, one column per side. Always visible."""
    h, w = frame.shape[:2]

    card_w = 230
    margin = 14
    pad = 12

    if side == "left":
        x0 = margin
        sections = [
            ("ACTIVAR", None, None),
            (None, "2 aplausos", "ON"),
            (None, "spread 2m 1s", "ON"),
            (None, "fist 2m", "OFF"),
            ("MEDIA", None, None),
            (None, "pinch", "play/pause"),
            (None, "peace L", "previous"),
            (None, "peace R", "next"),
            ("VOLUMEN", None, None),
            (None, "3 dedos", "vol +"),
            (None, "point (1)", "vol -"),
        ]
    else:
        x0 = w - card_w - margin
        sections = [
            ("VENTANAS", None, None),
            (None, "fist", "agarrar"),
            (None, "spread", "soltar"),
            (None, "mano arriba", ">> TV"),
            (None, "abajo-izq", ">> IZQ"),
            (None, "abajo-der", ">> DER"),
            (None, "rock 2m", "UNDO"),
            ("VOZ", None, None),
            (None, "despierta nexo", "ON"),
            (None, "descansa nexo", "OFF"),
            (None, "abre/escribe/busca", "comandos"),
            ("SISTEMA", None, None),
            (None, "rock L/R", "desktop"),
            (None, "ok hold", "cmd+tab"),
            (None, "apuntar 0.8s", "cursor"),
        ]

    y0 = 100
    row_h = 20
    card_h = 16 + row_h * len(sections)

    overlay = frame.copy()
    cv2.rectangle(overlay, (x0, y0), (x0 + card_w, y0 + card_h), (8, 8, 10), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    accent_x = x0 if side == "left" else x0 + card_w - 2
    cv2.rectangle(frame, (accent_x, y0), (accent_x + 2, y0 + card_h), COLOR_CYAN, -1)

    for i, (heading, left_text, right_text) in enumerate(sections):
        yy = y0 + 20 + i * row_h
        if heading is not None:
            cv2.putText(frame, heading, (x0 + pad, yy),
                        cv2.FONT_HERSHEY_DUPLEX, 0.48, COLOR_CYAN, 1, cv2.LINE_AA)
        else:
            cv2.putText(frame, left_text, (x0 + pad + 4, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.42, COLOR_WHITE, 1, cv2.LINE_AA)
            cv2.putText(frame, right_text, (x0 + pad + 120, yy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, COLOR_MUTED, 1, cv2.LINE_AA)


def draw_action_log(frame, log: list[tuple[float, str]]):
    if not log:
        return
    h, w = frame.shape[:2]
    pad_right = 20
    y_base = h - 40
    now = time.time()
    visible = log[-4:][::-1]
    for i, (ts, label) in enumerate(visible):
        age = now - ts
        if age > 8:
            continue
        alpha = max(0.2, 1.0 - age / 8)
        color = tuple(int(c * alpha) for c in COLOR_CYAN)
        yy = y_base - i * 20
        text = label[:40]
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0]
        x = w - pad_right - text_size[0]
        cv2.putText(frame, text, (x, yy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)


def draw_action_flash(frame, label: str | None, until: float):
    if not label or time.time() >= until:
        return
    h, w = frame.shape[:2]
    remaining = until - time.time()
    alpha = min(1.0, remaining / 0.35)

    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 1.6, 3)[0]
    tx = (w - text_size[0]) // 2
    ty = h // 2 + 10

    bar = frame.copy()
    cv2.rectangle(bar, (tx - 30, ty - 45), (tx + text_size[0] + 30, ty + 20), (20, 20, 24), -1)
    cv2.addWeighted(bar, 0.5 * alpha, frame, 1 - 0.5 * alpha, 0, frame)

    shadow = frame.copy()
    cv2.putText(shadow, label, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, (0, 0, 0), 6, cv2.LINE_AA)
    cv2.putText(shadow, label, (tx, ty),
                cv2.FONT_HERSHEY_DUPLEX, 1.6, COLOR_WHITE, 2, cv2.LINE_AA)
    cv2.addWeighted(shadow, alpha, frame, 1 - alpha, 0, frame)


def draw_drag_overlay(frame, hand_px: tuple[int, int], zone_label: str, trail_points: list[tuple[int, int]] | None = None):
    h, w = frame.shape[:2]

    # Iron Man style blue trail behind the hand
    if trail_points and len(trail_points) > 1:
        n = len(trail_points)
        for i in range(n - 1):
            alpha = (i + 1) / n
            thickness = max(2, int(alpha * 10))
            color = (int(255 * alpha), int(220 * alpha), int(120 * alpha))
            cv2.line(frame, trail_points[i], trail_points[i + 1], color, thickness, cv2.LINE_AA)

    cx, cy = hand_px
    overlay = frame.copy()
    for r, a in [(90, 0.1), (60, 0.2), (35, 0.35)]:
        cv2.circle(overlay, (cx, cy), r, (80, 200, 255), -1, cv2.LINE_AA)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)
    cv2.circle(frame, (cx, cy), 20, (120, 220, 255), 2, cv2.LINE_AA)
    cv2.circle(frame, (cx, cy), 8, (255, 255, 255), -1, cv2.LINE_AA)

    label = f">> {zone_label}"
    text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_DUPLEX, 0.8, 2)[0]
    tx = (w - text_size[0]) // 2
    cv2.putText(frame, label, (tx, h // 2 + 60),
                cv2.FONT_HERSHEY_DUPLEX, 0.8, (120, 220, 255), 2, cv2.LINE_AA)


def draw_permission_banner(frame):
    h, w = frame.shape[:2]
    banner = frame.copy()
    y0 = h - 95
    y1 = h - 55
    cv2.rectangle(banner, (0, y0), (w, y1), (20, 20, 40), -1)
    cv2.rectangle(banner, (0, y0), (w, y0 + 2), COLOR_RED, -1)
    cv2.addWeighted(banner, 0.75, frame, 0.25, 0, frame)
    cv2.putText(frame, "FALTA PERMISO: System Settings > Privacy > Accessibility > + agregar Terminal",
                (20, y0 + 28),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (180, 200, 255), 1, cv2.LINE_AA)
