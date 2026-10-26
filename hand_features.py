"""Deteksi tangan dan fitur tekukan jari."""

import os
import time

import cv2
import mediapipe as mp
import numpy as np

WRIST = 0
MIDDLE_MCP = 9
PINKY_MCP = 17
THUMB_TIP = 4
FINGER_TIPS = [8, 12, 16, 20]  # telunjuk, tengah, manis, kelingking
# (pangkal, tengah, ujung) untuk sudut: jempol MCP-IP-TIP, jari lain MCP-PIP-DIP
ANGLE_TRIPLETS = [(2, 3, 4), (5, 6, 7), (9, 10, 11), (13, 14, 15), (17, 18, 19)]
FEATURE_METHODS = ("jarak", "sudut")
LOST_RESET_S = 1.0  # lupakan tangan terakhir jika hilang selama ini


def _dist(p, q):
    return float(np.hypot(p[0] - q[0], p[1] - q[1]))


def _angle_deg(a, b, c):
    """Sudut di titik b (derajat) antara garis b->a dan b->c."""
    v1, v2 = np.asarray(a) - np.asarray(b), np.asarray(c) - np.asarray(b)
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return None
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))


def palm_size_px(hand_landmarks, width, height):
    """Ukuran telapak (piksel): jarak pergelangan ke pangkal jari tengah."""
    lm = hand_landmarks.landmark
    p0 = (lm[WRIST].x * width, lm[WRIST].y * height)
    p9 = (lm[MIDDLE_MCP].x * width, lm[MIDDLE_MCP].y * height)
    return _dist(p0, p9)


def compute_features(hand_landmarks, width, height, method="jarak"):
    """Mengembalikan array 5 nilai (jempol..kelingking), atau None jika gagal."""
    lm = hand_landmarks.landmark
    if method == "sudut":
        pts = np.array([[p.x * width, p.y * height, p.z * width] for p in lm])
        feats = []
        for a, b, c in ANGLE_TRIPLETS:
            ang = _angle_deg(pts[a], pts[b], pts[c])
            if ang is None:
                return None
            feats.append(ang)
        return np.array(feats)

    pts = np.array([[p.x * width, p.y * height] for p in lm])
    palm = _dist(pts[WRIST], pts[MIDDLE_MCP])
    if palm < 1e-6:
        return None
    feats = [_dist(pts[THUMB_TIP], pts[PINKY_MCP]) / palm]
    for tip in FINGER_TIPS:
        feats.append(_dist(pts[tip], pts[WRIST]) / palm)
    return np.array(feats)


def _open_camera(index):
    if os.name == "nt":  # backend DirectShow lebih stabil di Windows
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


class HandTracker:
    def __init__(self, camera_index=0, det_conf=0.6, track_conf=0.6, method="jarak"):
        if method not in FEATURE_METHODS:
            raise ValueError(f"method harus salah satu dari {FEATURE_METHODS}")
        self.method = method
        self.cap = _open_camera(camera_index)
        self._hands_mod = mp.solutions.hands
        self._drawer = mp.solutions.drawing_utils
        self.hands = self._hands_mod.Hands(
            static_image_mode=False,
            max_num_hands=2,
            min_detection_confidence=det_conf,
            min_tracking_confidence=track_conf,
        )
        self.prev_wrist = None
        self.last_seen = 0.0
        self.n_hands = 0        # jumlah tangan terlihat pada frame terakhir
        self.palm_ratio = 0.0   # ukuran telapak / lebar frame pada frame terakhir

    def is_open(self):
        return self.cap.isOpened()

    def read(self):
        """Membaca satu frame (dicerminkan). None jika kamera gagal."""
        ok, frame = self.cap.read()
        if not ok:
            return None
        return cv2.flip(frame, 1)

    def _select(self, result, w, h):
        hands = result.multi_hand_landmarks
        if len(hands) == 1:
            return hands[0]
        if self.prev_wrist is not None:
            d = []
            for hl in hands:
                p = np.array([hl.landmark[WRIST].x * w, hl.landmark[WRIST].y * h])
                d.append(_dist(p, self.prev_wrist))
            return hands[int(np.argmin(d))]
        scores = [hd.classification[0].score for hd in result.multi_handedness]
        return hands[int(np.argmax(scores))]

    def detect(self, frame_bgr):
        """Mengembalikan (fitur 5 jari, landmark) atau (None, None)."""
        now = time.time()
        if self.prev_wrist is not None and now - self.last_seen > LOST_RESET_S:
            self.prev_wrist = None
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        rgb.flags.writeable = False
        result = self.hands.process(rgb)
        if not result.multi_hand_landmarks:
            self.n_hands = 0
            return None, None
        h, w = frame_bgr.shape[:2]
        self.n_hands = len(result.multi_hand_landmarks)
        lm = self._select(result, w, h)
        self.prev_wrist = np.array([lm.landmark[WRIST].x * w, lm.landmark[WRIST].y * h])
        self.last_seen = now
        self.palm_ratio = palm_size_px(lm, w, h) / float(w)
        return compute_features(lm, w, h, self.method), lm

    def draw(self, frame_bgr, landmarks):
        self._drawer.draw_landmarks(
            frame_bgr, landmarks, self._hands_mod.HAND_CONNECTIONS
        )

    def release(self):
        self.cap.release()
        self.hands.close()