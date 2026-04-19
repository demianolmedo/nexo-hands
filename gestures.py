"""Gesture detection using MediaPipe Hands. Same gesture vocabulary as dj-turbo."""
import math


def _distance(a, b):
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2)


def detect_gesture(landmarks) -> str | None:
    thumb_tip = landmarks.landmark[4]
    index_tip = landmarks.landmark[8]

    thumb_up = landmarks.landmark[4].y < landmarks.landmark[3].y
    index_up = landmarks.landmark[8].y < landmarks.landmark[6].y
    middle_up = landmarks.landmark[12].y < landmarks.landmark[10].y
    ring_up = landmarks.landmark[16].y < landmarks.landmark[14].y
    pinky_up = landmarks.landmark[20].y < landmarks.landmark[18].y

    distance_thumb_index = _distance(thumb_tip, index_tip)

    # FIST — relaxed distance threshold (0.08 → 0.1) but keep all fingers down requirement
    if not any([index_up, middle_up, ring_up, pinky_up]) and distance_thumb_index < 0.10:
        return 'fist'

    if index_up and pinky_up and not middle_up and not ring_up:
        return 'rock'

    if index_up and middle_up and ring_up and not pinky_up:
        return 'three'

    if index_up and not middle_up and not ring_up and not pinky_up and distance_thumb_index > 0.1:
        return 'point'

    if index_up and middle_up and not ring_up and not pinky_up:
        return 'peace'

    if all([thumb_up, index_up, middle_up, ring_up, pinky_up]):
        return 'spread'

    if thumb_up and not any([middle_up, ring_up, pinky_up]) and distance_thumb_index > 0.2:
        return 'ok'

    if distance_thumb_index < 0.08 and not any([middle_up, ring_up, pinky_up]):
        return 'pinch'

    if distance_thumb_index < 0.05:
        return "great"

    return None
