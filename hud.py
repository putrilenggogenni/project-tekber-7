"""Penggambar teks di layar (HUD): satu lapis di atas kotak latar gelap."""

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
WHITE = (255, 255, 255)
YELLOW = (0, 255, 255)
GREEN = (0, 255, 0)
RED = (60, 60, 255)


def _thickness(scale):
    return 2 if scale >= 0.7 else 1


def put_text(frame, text, y, color=WHITE, scale=0.7, x=10):
    """Menulis teks dengan kotak latar gelap. y = garis dasar teks."""
    th = _thickness(scale)
    (w, h), base = cv2.getTextSize(text, FONT, scale, th)
    pad = 6
    x1, y1 = max(x - pad, 0), max(y - h - pad, 0)
    x2 = min(x + w + pad, frame.shape[1])
    y2 = min(y + base + pad, frame.shape[0])
    roi = frame[y1:y2, x1:x2]
    if roi.size:
        frame[y1:y2, x1:x2] = (roi * 0.3).astype(np.uint8)
    cv2.putText(frame, text, (x, y), FONT, scale, color, th, cv2.LINE_AA)


def draw_hud(frame, lines, scale=0.7, line_h=36, x=10, y0=30):
    """Menggambar daftar (teks, warna) bersusun ke bawah.
    Teks yang lebih lebar dari layar dipecah ke baris berikutnya."""
    th = _thickness(scale)
    max_w = frame.shape[1] - x - 15
    y = y0
    for text, color in lines:
        current = ""
        for word in text.split(" "):
            trial = (current + " " + word).strip()
            width = cv2.getTextSize(trial, FONT, scale, th)[0][0]
            if width > max_w and current:
                put_text(frame, current, y, color, scale, x)
                y += line_h
                current = word
            else:
                current = trial
        if current:
            put_text(frame, current, y, color, scale, x)
            y += line_h