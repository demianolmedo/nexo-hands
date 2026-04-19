"""Gesture-control main loop: camera → gestures → OS actions."""
import sys
import time
import cv2
import mediapipe as mp

from gestures import detect_gesture
import media
import windows
import audio as audio_mod
from hud import (
    draw_status, draw_drag_overlay,
    draw_side_legend, draw_action_log, draw_permission_banner,
    draw_action_flash,
)


# -------------------- state --------------------
ACTIVE = False
ATTENTION_UNTIL = 0.0
IDLE_TIMEOUT_SEC = 30.0
ACTION_COOLDOWN = 0.9
HOLD_THRESHOLD_SEC = 1.0
POST_GRAB_GRACE_SEC = 1.2  # ignore other gestures briefly after a grab/release
POINTER_HANDOFF_HOLD_SEC = 0.8     # aim at another screen this long → teleport
POINTER_HANDOFF_COOLDOWN_SEC = 1.5  # min gap between teleports (soft)
STATE_CHANGE_COOLDOWN_SEC = 3.0    # min gap between activate/deactivate transitions
CLAP_BLACKOUT_AFTER_ACTIVATE_SEC = 40.0  # ignore claps for 40s after activation (music playing)

last_action_ts: dict[str, float] = {}
flash_label: str | None = None
flash_until = 0.0
action_log: list[tuple[float, str]] = []  # (ts, label)

both_spread_since: float | None = None
both_fist_since: float | None = None
both_rock_since: float | None = None
ok_hold_since: float | None = None
grab_released_at: float = 0.0
pointer_aiming_display: int | None = None
pointer_aiming_since: float = 0.0
pointer_last_teleport_at: float = 0.0
pointer_last_teleport_display: int | None = None

# Undo: remember last window move
last_move: dict | None = None  # {"window_el", "pid", "prev_display"}

# Debounce state transitions (prevent activate→deactivate→activate loops)
last_state_change_ts: float = 0.0
last_activate_ts: float = 0.0

# Grab state
grabbed_window = None
grabbed_pid: int | None = None
grabbed_app_name: str | None = None
grab_hand_side: str | None = None  # "L" or "R"
grab_active = False
grabbed_origin_display: int | None = None  # display where the fist happened
grab_lost_frames: int = 0  # tolerance for detector flicker during drag
GRAB_LOST_THRESHOLD = 25   # ~0.8s at 30fps — auto-release if no fist/spread seen

# Hand trail for throw velocity (option A) + visual trail (option B)
hand_trail: list[tuple[float, float, float]] = []  # (ts, x_norm, y_norm)
HAND_TRAIL_WINDOW_SEC = 0.35
THROW_SPEED_THRESHOLD = 0.6  # normalized units per second

HAS_ACCESSIBILITY = False


def flash(label: str, duration: float = 1.3):
    global flash_label, flash_until
    flash_label = label
    flash_until = time.time() + duration
    action_log.append((time.time(), label))
    # keep last 4
    while len(action_log) > 4:
        action_log.pop(0)
    print(f"[action] {label}")


def fire_action(key: str, fn, label: str, cooldown: float = ACTION_COOLDOWN):
    now = time.time()
    if now - last_action_ts.get(key, 0.0) < cooldown:
        return
    last_action_ts[key] = now
    try:
        fn()
    except Exception as e:
        flash(f"ERROR: {e}")
        return
    flash(label)


def activate():
    global ACTIVE, ATTENTION_UNTIL, last_state_change_ts, last_activate_ts
    now = time.time()
    if ACTIVE:
        return
    if now - last_state_change_ts < STATE_CHANGE_COOLDOWN_SEC:
        return
    ACTIVE = True
    ATTENTION_UNTIL = now + IDLE_TIMEOUT_SEC
    last_state_change_ts = now
    last_activate_ts = now
    flash("GESTOS ACTIVOS")
    audio_mod.play_welcome_sequence()


def deactivate():
    global ACTIVE, last_state_change_ts
    now = time.time()
    if not ACTIVE:
        return
    if now - last_state_change_ts < STATE_CHANGE_COOLDOWN_SEC:
        return
    ACTIVE = False
    last_state_change_ts = now
    flash("GESTOS OFF")
    audio_mod.play_seeya()


def refresh_attention():
    global ATTENTION_UNTIL
    ATTENTION_UNTIL = time.time() + IDLE_TIMEOUT_SEC


def main():
    global ACTIVE, both_spread_since, both_fist_since, both_rock_since, ok_hold_since
    global grabbed_window, grabbed_pid, grabbed_app_name, grab_active, grab_hand_side, grabbed_origin_display, grab_lost_frames
    global HAS_ACCESSIBILITY, grab_released_at
    global pointer_aiming_display, pointer_aiming_since, pointer_last_teleport_at, pointer_last_teleport_display
    global last_move, last_state_change_ts, last_activate_ts, hand_trail

    cap = None
    for cam_idx in range(4):
        test = cv2.VideoCapture(cam_idx)
        if test.isOpened():
            ok, _ = test.read()
            if ok:
                cap = test
                print(f"[camera] using device index {cam_idx}")
                break
            test.release()
        else:
            test.release()
    if cap is None:
        print("Error: no working camera found (tried indices 0-3)")
        sys.exit(1)

    cv2.namedWindow('Gesture Control', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('Gesture Control', 960, 540)
    try:
        cv2.setWindowProperty('Gesture Control', cv2.WND_PROP_TOPMOST, 1)
    except Exception:
        pass

    def _force_topmost():
        """Backup: lift our NSWindow to floating level (survives across foreground-switch)."""
        try:
            from AppKit import NSApp
            for ww in NSApp.windows():
                try:
                    if 'Gesture Control' in str(ww.title()):
                        ww.setLevel_(3)  # NSFloatingWindowLevel
                except Exception:
                    continue
        except Exception:
            pass

    last_topmost_refresh = 0.0

    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils
    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.5,
    )

    # Permission check (runs once)
    HAS_ACCESSIBILITY = windows.check_accessibility_permission()
    print(f"[startup] Accessibility permission: {HAS_ACCESSIBILITY}")

    last_perm_check = time.time()

    # Double-clap activator (microphone) — alternative to spread×2
    def _on_double_clap():
        # Ignore claps during activation music playback to avoid self-triggering
        if time.time() - last_activate_ts < CLAP_BLACKOUT_AFTER_ACTIVATE_SEC:
            return
        if not ACTIVE:
            activate()

    clap_det = audio_mod.ClapDetector(threshold=0.18, on_double_clap=_on_double_clap)
    try:
        clap_det.start()
        print("[startup] clap detector running")
    except Exception as e:
        print(f"[startup] clap detector failed: {e}")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        left_hand = None
        right_hand = None
        if result.multi_hand_landmarks:
            for lm in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)
                if lm.landmark[0].x <= 0.5:
                    left_hand = lm
                else:
                    right_hand = lm

        gesture_l = detect_gesture(left_hand) if left_hand else None
        gesture_r = detect_gesture(right_hand) if right_hand else None

        # Recheck accessibility every 5s in case user granted it mid-run
        if time.time() - last_perm_check > 5:
            HAS_ACCESSIBILITY = windows.check_accessibility_permission()
            last_perm_check = time.time()

        # -------- ATTENTION STATE --------
        both_spread = (gesture_l == "spread" and gesture_r == "spread")
        both_fist = (gesture_l == "fist" and gesture_r == "fist")
        both_rock = (gesture_l == "rock" and gesture_r == "rock")

        if both_spread and not ACTIVE:
            if both_spread_since is None:
                both_spread_since = time.time()
            elif time.time() - both_spread_since >= HOLD_THRESHOLD_SEC:
                activate()
                both_spread_since = None
        else:
            both_spread_since = None

        if both_fist and ACTIVE:
            if both_fist_since is None:
                both_fist_since = time.time()
            elif time.time() - both_fist_since >= 0.4:
                deactivate()
                both_fist_since = None
                grab_active = False
                grabbed_window = None
                grabbed_app_name = None
                grab_hand_side = None
        else:
            both_fist_since = None

        if ACTIVE and time.time() > ATTENTION_UNTIL and not grab_active:
            deactivate()

        # -------- UNDO: rock with BOTH hands --------
        if ACTIVE and both_rock and not grab_active:
            if both_rock_since is None:
                both_rock_since = time.time()
            elif time.time() - both_rock_since >= 0.4 and last_move is not None:
                now = time.time()
                if now - last_action_ts.get("undo", 0) >= 2.0:
                    last_action_ts["undo"] = now
                    # Move the window back to its origin display
                    windows.move_window_to_display(
                        last_move["window_el"], last_move["pid"], last_move["prev_display"], maximize=True
                    )
                    zone = windows.zone_label_for_display(last_move["prev_display"], windows.list_displays())
                    flash(f"UNDO >> {zone}")
                    last_move = None
                    refresh_attention()
                    both_rock_since = None
        else:
            both_rock_since = None

        # -------- ACTIONS (only when ACTIVE) --------
        if ACTIVE:
            # GRAB detection: single-hand fist (not both)
            # Cross-screen grab: pick the primary window of the display where the hand is pointing.
            if not grab_active and (gesture_l == "fist") != (gesture_r == "fist"):
                grab_side = "L" if gesture_l == "fist" else "R"
                grab_hand = left_hand if grab_side == "L" else right_hand
                if grab_hand is not None:
                    hx = float(grab_hand.landmark[0].x)
                    hy = float(grab_hand.landmark[0].y)
                else:
                    hx, hy = 0.5, 0.5
                displays = windows.list_displays()
                origin_display = windows.classify_hand_position_to_display(hx, hy, displays)

                # Try cross-screen pick first
                primary = windows.get_primary_window_on_display(origin_display) if origin_display is not None else None
                if primary is not None:
                    grabbed_pid, grabbed_window, grabbed_app_name = primary
                    grab_active = True
                    grab_hand_side = grab_side
                    grabbed_origin_display = origin_display
                    refresh_attention()
                    origin_zone = windows.zone_label_for_display(origin_display, displays)
                    flash(f"GRAB {grabbed_app_name} [{origin_zone}]")
                else:
                    # Fallback to frontmost global if nothing found on that screen
                    win = windows.get_frontmost_window_info()
                    if win is not None:
                        grabbed_pid, grabbed_window = win
                        from AppKit import NSWorkspace
                        ws = NSWorkspace.sharedWorkspace()
                        for app in ws.runningApplications():
                            if int(app.processIdentifier()) == grabbed_pid:
                                grabbed_app_name = str(app.localizedName())
                                break
                        grab_active = True
                        grab_hand_side = grab_side
                        refresh_attention()
                        flash(f"GRAB {grabbed_app_name}")
                    else:
                        flash("GRAB FAIL (accessibility?)")

            if grab_active:
                grab_hand = left_hand if grab_hand_side == "L" else right_hand
                grab_gesture = gesture_l if grab_hand_side == "L" else gesture_r

                # Update hand trail (for throw velocity + visual trail)
                if grab_hand is not None:
                    now_ts = time.time()
                    hx_now = float(grab_hand.landmark[0].x)
                    hy_now = float(grab_hand.landmark[0].y)
                    hand_trail.append((now_ts, hx_now, hy_now))
                    while hand_trail and hand_trail[0][0] < now_ts - HAND_TRAIL_WINDOW_SEC:
                        hand_trail.pop(0)

                # Tolerance: if the detector flickers (neither fist nor spread), keep the grab
                # alive for up to GRAB_LOST_THRESHOLD frames. This fixes the mid-drag "GRAB FAIL"
                # when the hand is in transit and briefly classifies as something else.
                if grab_gesture == "fist":
                    grab_lost_frames = 0
                elif grab_gesture != "spread":
                    grab_lost_frames += 1
                    if grab_lost_frames < GRAB_LOST_THRESHOLD:
                        grab_gesture = "fist"  # pretend we're still gripping

                if grab_gesture == "spread":
                    if grab_hand is not None:
                        hx = float(grab_hand.landmark[0].x)
                        hy = float(grab_hand.landmark[0].y)
                    else:
                        hx, hy = 0.5, 0.5

                    # Throw velocity — if the hand moved fast in a direction just before spread,
                    # extrapolate that direction to pick the target zone.
                    if len(hand_trail) >= 3:
                        t0, x0, y0 = hand_trail[0]
                        t1, x1, y1 = hand_trail[-1]
                        dt = max(0.05, t1 - t0)
                        vx = (x1 - x0) / dt
                        vy = (y1 - y0) / dt
                        speed = (vx ** 2 + vy ** 2) ** 0.5
                        if speed > THROW_SPEED_THRESHOLD:
                            hx = max(0.0, min(1.0, x1 + vx * 0.25))
                            hy = max(0.0, min(1.0, y1 + vy * 0.25))

                    displays = windows.list_displays()
                    target = windows.classify_hand_position_to_display(hx, hy, displays)
                    if target is not None and grabbed_window is not None and grabbed_pid is not None:
                        app = windows.move_window_to_display(grabbed_window, grabbed_pid, target, maximize=True)
                        zone = windows.zone_label_for_display(target, displays)
                        flash(f">> {zone}")
                        if grabbed_origin_display is not None and grabbed_origin_display != target:
                            last_move = {
                                "window_el": grabbed_window,
                                "pid": grabbed_pid,
                                "prev_display": grabbed_origin_display,
                            }
                    grab_active = False
                    grabbed_window = None
                    grabbed_app_name = None
                    grab_hand_side = None
                    grabbed_origin_display = None
                    grab_released_at = time.time()
                    hand_trail.clear()
                    refresh_attention()
                elif grab_gesture != "fist" and grab_lost_frames >= GRAB_LOST_THRESHOLD:
                    # Only drop the grab after prolonged loss of tracking
                    grab_active = False
                    grabbed_window = None
                    grabbed_app_name = None
                    grab_hand_side = None
                    grabbed_origin_display = None
                    grab_released_at = time.time()
                    grab_lost_frames = 0
            else:
                # ---- POINTER HANDOFF ----
                # Smart cooldown: don't teleport to the SAME display we just teleported to,
                # unless the hand first exited that zone and came back (user re-aiming).
                active_hand = left_hand or right_hand
                if active_hand is not None and time.time() - pointer_last_teleport_at > POINTER_HANDOFF_COOLDOWN_SEC:
                    hx = float(active_hand.landmark[0].x)
                    hy = float(active_hand.landmark[0].y)
                    displays = windows.list_displays()
                    aim_display = windows.classify_hand_position_to_display(hx, hy, displays)
                    cursor_display = windows.get_cursor_display_index(displays)

                    # Skip if aiming at same display as cursor OR the last teleport target (unless hand left)
                    same_as_cursor = aim_display == cursor_display
                    same_as_last = aim_display == pointer_last_teleport_display

                    if aim_display is not None and not same_as_cursor and not same_as_last:
                        if pointer_aiming_display == aim_display:
                            if time.time() - pointer_aiming_since >= POINTER_HANDOFF_HOLD_SEC:
                                if windows.teleport_cursor_to_display(aim_display):
                                    zone = windows.zone_label_for_display(aim_display, displays)
                                    flash(f"CURSOR >> {zone}")
                                    pointer_last_teleport_at = time.time()
                                    pointer_last_teleport_display = aim_display
                                pointer_aiming_display = None
                        else:
                            pointer_aiming_display = aim_display
                            pointer_aiming_since = time.time()
                    else:
                        pointer_aiming_display = None
                        # If hand left the last-teleport-target zone, reset it so we can teleport back
                        if aim_display is not None and aim_display != pointer_last_teleport_display:
                            pointer_last_teleport_display = None
                else:
                    pointer_aiming_display = None

                # Post-grab grace period: ignore all actions for a moment after releasing a window
                in_grace = time.time() - grab_released_at < POST_GRAB_GRACE_SEC

                if in_grace:
                    # Still draw HUD but skip action dispatch
                    pass
                # Non-drag gestures (only outside grace window)
                # pinch single hand → play/pause (avoids conflict with spread=release)
                if not in_grace and (gesture_l == "pinch" or gesture_r == "pinch"):
                    fire_action("play_pause", media.play_pause, "PLAY / PAUSE")
                    refresh_attention()

                if not in_grace and gesture_l == "peace":
                    fire_action("prev", media.previous_track, "PREV TRACK")
                    refresh_attention()
                if not in_grace and gesture_r == "peace":
                    fire_action("next", media.next_track, "NEXT TRACK")
                    refresh_attention()

                # Only fire desktop switch with single-hand rock (two-hand rock = undo)
                if not in_grace and gesture_l == "rock" and not both_rock:
                    fire_action("desk_left", media.desktop_left, "DESKTOP LEFT")
                    refresh_attention()
                if not in_grace and gesture_r == "rock" and not both_rock:
                    fire_action("desk_right", media.desktop_right, "DESKTOP RIGHT")
                    refresh_attention()

                now = time.time()
                if not in_grace and (gesture_l == "three" or gesture_r == "three"):
                    if now - last_action_ts.get("vol_up", 0) >= 0.12:
                        media.volume_up(step=2)
                        last_action_ts["vol_up"] = now
                        refresh_attention()
                if not in_grace and (gesture_l == "point" or gesture_r == "point"):
                    if now - last_action_ts.get("vol_down", 0) >= 0.12:
                        media.volume_down(step=2)
                        last_action_ts["vol_down"] = now
                        refresh_attention()

                if not in_grace and (gesture_l == "ok" or gesture_r == "ok"):
                    if ok_hold_since is None:
                        ok_hold_since = time.time()
                    elif time.time() - ok_hold_since >= HOLD_THRESHOLD_SEC:
                        fire_action("cmd_tab", media.switch_next_app, "CMD + TAB")
                        ok_hold_since = None
                        refresh_attention()
                else:
                    ok_hold_since = None

        # -------- DRAW --------
        draw_status(frame, ACTIVE, ATTENTION_UNTIL, gesture_l, gesture_r,
                    grabbed_app_name if grab_active else None)

        # Side legends (always visible)
        draw_side_legend(frame, "left")
        draw_side_legend(frame, "right")

        # Action log (right side, below legend)
        draw_action_log(frame, action_log)

        # Drag overlay
        if grab_active:
            grab_hand = left_hand if grab_hand_side == "L" else right_hand
            if grab_hand is not None:
                hx = int(grab_hand.landmark[0].x * w)
                hy = int(grab_hand.landmark[0].y * h)
                displays = windows.list_displays()
                target = windows.classify_hand_position_to_display(
                    grab_hand.landmark[0].x, grab_hand.landmark[0].y, displays
                )
                zone = windows.zone_label_for_display(target, displays) if target is not None else "?"
                # Convert normalized trail to pixel coords for visual rendering
                trail_px = [(int(tx * w), int(ty * h)) for _, tx, ty in hand_trail]
                draw_drag_overlay(frame, (hx, hy), zone, trail_px)

        # Flash in the center
        draw_action_flash(frame, flash_label, flash_until)

        # Permission banner if missing
        if not HAS_ACCESSIBILITY:
            draw_permission_banner(frame)

        cv2.imshow('Gesture Control', frame)

        # Refresh topmost level every 2s (may lose it when other apps activate)
        if time.time() - last_topmost_refresh > 2.0:
            _force_topmost()
            try:
                cv2.setWindowProperty('Gesture Control', cv2.WND_PROP_TOPMOST, 1)
            except Exception:
                pass
            last_topmost_refresh = time.time()

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    try:
        clap_det.stop()
    except Exception:
        pass
    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
