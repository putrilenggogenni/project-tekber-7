"""Deteksi tangan dan fitur tekukan jari.

Fitur per jari = jarak ujung jari ke pergelangan dibagi ukuran telapak
(jarak pergelangan ke pangkal jari tengah). Nilai BESAR = jari lurus/terbuka,
nilai KECIL = jari ditekuk. Untuk jempol dipakai jarak ujung jempol ke
pangkal kelingking.
Ini estimasi relatif, bukan sudut sendi aktual (kamera 2D tidak mengukur kedalaman).
"""

import cv2
import mediapipe as mp
import numpy as np

WRIST = 0
MIDDLE_MCP = 9
PINKY_MCP = 17
THUMB_TIP = 4
FINGER_TIPS = [8, 12, 16, 20]  # telunjuk, tengah, manis, kelingking


def _dist(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def compute_features(hand_landmarks, width, height):
    """Mengembalikan array 5 nilai (jempol..kelingking), atau None jika gagal."""
    pts = np.array(
        [[lm.x * width, lm.y * height] for lm in hand_landmarks.landmark]
    )
    palm = _dist(pts[WRIST], pts[MIDDLE_MCP])
    if palm < 1e-6:
        return None
    feats = [_dist(pts[THUMB_TIP], pts[PINKY_MCP]) / palm]
    for tip in FINGER_TIPS:
        feats.append(_dist(pts[tip], pts[WRIST]) / palm)
    return np.array(feats)


class HandTracker:
    def __init__(self, camera_index=0, det_conf=0.6, track_conf=0.6):
        self.cap = cv2.VideoCapture(camera_index, cv2.CAP_DSHOW)
        self._hands_mod = mp.solutions.hands
        self._drawer = mp.solutions.drawing_utils
        self.hands = self._hands_mod.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )

    def read(self):
        """Membaca satu frame (dicerminkan). None jika kamera gagal."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv2.flip(frame, 1)

    def detect(self, frame_bgr):
        """Mengembalikan (fitur 5 jari, landmark) atau (None, None)."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            return None, None
        lm = result.multi_hand_landmarks[0]
        h, w = frame_bgr.shape[:2]
        return compute_features(lm, w, h), lm

    def draw(self, frame_bgr, landmarks):
        self._drawer.draw_landmarks(
            frame_bgr, landmarks, self._hands_mod.HAND_CONNECTIONS
        )

    def release(self):
        self.cap.release()
        self.hands.close()
