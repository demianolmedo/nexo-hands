"""macOS window manager via pyobjc + Accessibility API. Move frontmost window between displays."""
from AppKit import NSScreen, NSWorkspace, NSRunningApplication
from ApplicationServices import (
    AXUIElementCreateApplication,
    AXUIElementCopyAttributeValue,
    AXUIElementSetAttributeValue,
    AXValueCreate,
    kAXValueCGPointType,
    kAXValueCGSizeType,
)
from Quartz import (
    CGPoint, CGSize,
    CGWindowListCopyWindowInfo,
    kCGWindowListOptionOnScreenOnly,
    kCGNullWindowID,
    CGWarpMouseCursorPosition,
    CGEventSourceCreate,
    kCGEventSourceStateCombinedSessionState,
    CGEventCreateMouseEvent,
    CGEventPost,
    kCGHIDEventTap,
    kCGEventMouseMoved,
)


def list_displays() -> list[dict]:
    """Return list of displays with frame (x, y, w, h) in global coords (bottom-left origin).
    Display index 0 is the main display."""
    screens = NSScreen.screens()
    out = []
    for i, s in enumerate(screens):
        f = s.frame()
        out.append({
            "index": i,
            "x": float(f.origin.x),
            "y": float(f.origin.y),
            "w": float(f.size.width),
            "h": float(f.size.height),
            "is_main": bool(s == NSScreen.mainScreen()),
        })
    return out


def get_frontmost_app_pid() -> int | None:
    ws = NSWorkspace.sharedWorkspace()
    app = ws.frontmostApplication()
    if app is None:
        return None
    return int(app.processIdentifier())


def check_accessibility_permission() -> bool:
    """Return True if process has Accessibility permission. Without it, AX calls fail silently."""
    pid = get_frontmost_app_pid()
    if pid is None:
        return False
    app_el = AXUIElementCreateApplication(pid)
    err, val = AXUIElementCopyAttributeValue(app_el, "AXFocusedWindow", None)
    return err == 0 and val is not None


def get_frontmost_window_info() -> tuple[int, object] | None:
    """Return (pid, AXUIElement of the frontmost window) or None if no accessible app."""
    pid = get_frontmost_app_pid()
    if pid is None:
        return None
    app_el = AXUIElementCreateApplication(pid)
    err, focused = AXUIElementCopyAttributeValue(app_el, "AXFocusedWindow", None)
    if err != 0 or focused is None:
        return None
    return pid, focused


def set_window_frame(window_el, x: float, y: float, w: float, h: float):
    """Move and resize window to the given global coords."""
    pos = AXValueCreate(kAXValueCGPointType, CGPoint(x, y))
    size = AXValueCreate(kAXValueCGSizeType, CGSize(w, h))
    AXUIElementSetAttributeValue(window_el, "AXPosition", pos)
    AXUIElementSetAttributeValue(window_el, "AXSize", size)


def move_window_to_display(window_el, pid: int, display_index: int, maximize: bool = True) -> str | None:
    """Move a specific AX window element to a target display."""
    displays = list_displays()
    rect = _display_cg_rect(display_index, displays)
    if rect is None:
        return None
    ax_x, ax_y, tw, th = rect
    target = displays[display_index]

    if maximize:
        margin_top = 28 if target.get("is_main") else 0
        set_window_frame(window_el, ax_x, ax_y + margin_top, tw, th - margin_top)
    else:
        set_window_frame(window_el, ax_x + 40, ax_y + 40, 0, 0)

    ws = NSWorkspace.sharedWorkspace()
    for app in ws.runningApplications():
        if int(app.processIdentifier()) == pid:
            return str(app.localizedName())
    return "Unknown"


def move_frontmost_to_display(display_index: int, maximize: bool = True) -> str | None:
    """Move frontmost window to the given display. Returns app name or None on failure."""
    win = get_frontmost_window_info()
    if win is None:
        return None
    pid, window_el = win
    return move_window_to_display(window_el, pid, display_index, maximize)


def _zone_indexes(displays: list[dict]) -> dict:
    """Map display indices to physical zones (top / bottom-left / bottom-right).

    NSScreen origin is bottom-left. Displays with the highest Y are physically above.
    Among the lowest, smallest X is to the physical left.
    """
    if not displays:
        return {}
    if len(displays) == 1:
        return {"bl": 0, "br": 0, "top": 0}
    # Find top (highest y)
    top_idx = max(range(len(displays)), key=lambda i: displays[i]["y"])
    top_y = displays[top_idx]["y"]
    bottoms = [i for i in range(len(displays)) if i != top_idx and displays[i]["y"] < top_y]
    if not bottoms:
        bottoms = [i for i in range(len(displays)) if i != top_idx]
    if len(bottoms) >= 2:
        bl_idx = min(bottoms, key=lambda i: displays[i]["x"])
        br_idx = max(bottoms, key=lambda i: displays[i]["x"])
    else:
        bl_idx = bottoms[0]
        br_idx = bottoms[0]
    return {"top": top_idx, "bl": bl_idx, "br": br_idx}


def classify_hand_position_to_display(hand_x_norm: float, hand_y_norm: float, displays: list[dict]) -> int | None:
    """Map normalized hand coords (x,y in [0,1], origin top-left in camera frame) to a display index.

    Zones (3-band):
    - Top third of frame (y < 0.33) → TV (top display)
    - Middle/bottom, x < 0.5 → bottom-left display
    - Middle/bottom, x >= 0.5 → bottom-right display
    """
    zones = _zone_indexes(displays)
    if not zones:
        return None
    if hand_y_norm < 0.33:
        return zones["top"]
    return zones["bl"] if hand_x_norm < 0.5 else zones["br"]


def _display_cg_rect(display_index: int, displays: list[dict]) -> tuple[float, float, float, float] | None:
    """Return (x, y, w, h) of a display in CG coords (top-left origin, main display at (0,0))."""
    if display_index < 0 or display_index >= len(displays):
        return None
    d = displays[display_index]
    # Main display: height in NS = height in CG. CG origin is top-left of main.
    main = next((dd for dd in displays if dd.get("is_main")), displays[0])
    main_h = main["h"]
    # ns_y is bottom-left distance from main's bottom. CG y is top-left distance from main's top.
    cg_y = main_h - (d["y"] + d["h"])
    return (d["x"], cg_y, d["w"], d["h"])


def get_windows_on_display(display_index: int) -> list[dict]:
    """List windows overlapping the given display (sorted by z-order, front first)."""
    displays = list_displays()
    rect = _display_cg_rect(display_index, displays)
    if rect is None:
        return []
    dx, dy, dw, dh = rect

    infos = CGWindowListCopyWindowInfo(kCGWindowListOptionOnScreenOnly, kCGNullWindowID)
    out = []
    for w in infos:
        # skip non-user windows (menu bar extra, dock, wallpaper)
        layer = w.get("kCGWindowLayer", 0)
        if layer != 0:
            continue
        bounds = w.get("kCGWindowBounds")
        if not bounds:
            continue
        wx = float(bounds.get("X", 0))
        wy = float(bounds.get("Y", 0))
        ww = float(bounds.get("Width", 0))
        wh = float(bounds.get("Height", 0))
        if ww * wh < 10000:  # ignore tiny helpers
            continue
        # overlap check — require >50% of the window area to fall within the display
        overlap_x = max(0.0, min(wx + ww, dx + dw) - max(wx, dx))
        overlap_y = max(0.0, min(wy + wh, dy + dh) - max(wy, dy))
        if (overlap_x * overlap_y) < 0.5 * (ww * wh):
            continue
        out.append({
            "pid": int(w.get("kCGWindowOwnerPID", 0)),
            "owner": str(w.get("kCGWindowOwnerName", "") or ""),
            "name": str(w.get("kCGWindowName", "") or ""),
            "bounds": (wx, wy, ww, wh),
        })
    return out


def get_primary_window_on_display(display_index: int) -> tuple[int, object, str] | None:
    """Bring the primary (topmost) window of a display to front and return (pid, AXElement, app_name)."""
    import time as _t
    wins = get_windows_on_display(display_index)
    if not wins:
        return None
    top = wins[0]
    pid = top["pid"]
    app_name = top["owner"] or "Unknown"

    app = NSRunningApplication.runningApplicationWithProcessIdentifier_(pid)
    if app:
        app.activateWithOptions_(2)

    app_el = AXUIElementCreateApplication(pid)
    # Retry up to 6 times over 300ms — some apps are slow to register focus after activation
    for _ in range(6):
        err, focused = AXUIElementCopyAttributeValue(app_el, "AXFocusedWindow", None)
        if err == 0 and focused is not None:
            return pid, focused, app_name
        # Try generic windows list if AXFocusedWindow still None
        err2, wins_list = AXUIElementCopyAttributeValue(app_el, "AXWindows", None)
        if err2 == 0 and wins_list:
            return pid, wins_list[0], app_name
        _t.sleep(0.05)
    return None


def get_cursor_display_index(displays: list[dict] | None = None) -> int | None:
    """Return the display index where the mouse cursor currently lives."""
    from AppKit import NSEvent
    loc = NSEvent.mouseLocation()  # NSScreen coords (bottom-left)
    if displays is None:
        displays = list_displays()
    for d in displays:
        if d["x"] <= loc.x < d["x"] + d["w"] and d["y"] <= loc.y < d["y"] + d["h"]:
            return d["index"]
    return None


def teleport_cursor_to_display(display_index: int) -> bool:
    """Move mouse cursor to center of given display (CG coords)."""
    displays = list_displays()
    rect = _display_cg_rect(display_index, displays)
    if rect is None:
        return False
    cx = rect[0] + rect[2] / 2
    cy = rect[1] + rect[3] / 2
    CGWarpMouseCursorPosition(CGPoint(cx, cy))
    return True


def move_cursor_to_hand_position(hand_x_norm: float, hand_y_norm: float) -> bool:
    """Map normalized hand coords to a screen coord following the zone convention."""
    displays = list_displays()
    target = classify_hand_position_to_display(hand_x_norm, hand_y_norm, displays)
    if target is None:
        return False
    rect = _display_cg_rect(target, displays)
    if rect is None:
        return False
    dx, dy, dw, dh = rect

    # Map hand coord within its zone to local (0-1, 0-1), then to display rect
    if hand_y_norm < 0.33:
        # TV zone uses full x range
        local_x = hand_x_norm
        local_y = hand_y_norm / 0.33
    elif hand_x_norm < 0.5:
        local_x = hand_x_norm * 2
        local_y = (hand_y_norm - 0.33) / 0.67
    else:
        local_x = (hand_x_norm - 0.5) * 2
        local_y = (hand_y_norm - 0.33) / 0.67

    local_x = max(0.0, min(1.0, local_x))
    local_y = max(0.0, min(1.0, local_y))

    screen_x = dx + local_x * dw
    screen_y = dy + local_y * dh
    CGWarpMouseCursorPosition(CGPoint(screen_x, screen_y))
    return True


def zone_label_for_display(display_index: int, displays: list[dict]) -> str:
    zones = _zone_indexes(displays)
    if display_index == zones.get("top"):
        return "TV ARRIBA"
    if display_index == zones.get("bl"):
        return "IZQUIERDA ABAJO"
    if display_index == zones.get("br"):
        return "DERECHA ABAJO"
    return "?"
