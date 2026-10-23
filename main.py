"""Aplikasi utama: instrumen musik udara. Penanggung jawab: Anggota 3.

Tombol: C = kalibrasi, M = mode melodi, + / - = volume, Q = keluar dan simpan.
"""

import csv
import os
import time
from datetime import datetime

import cv2
import numpy as np

from audio import FINGER_NAMES, TonePlayer
from calibration import Calibrator
from hand_features import HandTracker
from temporal import TemporalFilter

CALIB_SECONDS = 8
MELODY = [0, 1, 2, 3, 4, 3, 2, 1]  # urutan jari target pada mode melodi
OUT_DIR = "hasil_sesi"
NO_HAND_RESET_S = 1.0


def put_text(frame, text, y, color=(255, 255, 255), scale=0.65):
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), 3, cv2.LINE_AA)
    cv2.putText(frame, text, (10, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, 1, cv2.LINE_AA)


def draw_bars(frame, levels):
    h, w = frame.shape[:2]
    bar_w, bar_h = 30, 100
    x0, y0 = w - 200, h - 130
    for i, lv in enumerate(levels):
        x = x0 + i * 38
        cv2.rectangle(frame, (x, y0), (x + bar_w, y0 + bar_h), (90, 90, 90), 1)
        fill = int(bar_h * float(lv))
        cv2.rectangle(frame, (x, y0 + bar_h - fill), (x + bar_w, y0 + bar_h), (0, 200, 0), -1)
        cv2.putText(frame, str(i + 1), (x + 9, y0 + bar_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


class SessionStats:
    def __init__(self):
        self.start = datetime.now()
        self.presses = np.zeros(5, dtype=int)
        self.rom = np.zeros(5)
        self.melody_ok = 0
        self.melody_bad = 0

    def save(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "ringkasan_sesi.csv")
        new_file = not os.path.exists(path)
        duration = (datetime.now() - self.start).total_seconds()
        total_melody = self.melody_ok + self.melody_bad
        acc = 100.0 * self.melody_ok / total_melody if total_melody else ""
        header = (["waktu_mulai", "durasi_detik"]
                  + [f"repetisi_{n.lower()}" for n in FINGER_NAMES]
                  + [f"rom_{n.lower()}" for n in FINGER_NAMES]
                  + ["melodi_benar", "melodi_salah", "akurasi_melodi_persen"])
        row = ([self.start.strftime("%Y-%m-%d %H:%M:%S"), round(duration, 1)]
               + [int(x) for x in self.presses]
               + [round(float(x), 3) for x in self.rom]
               + [self.melody_ok, self.melody_bad, acc])
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(header)
            writer.writerow(row)
        return path


def main():
    tracker = HandTracker()
    player = TonePlayer()
    calibrator = Calibrator()
    temporal = TemporalFilter()
    stats = SessionStats()

    calib = None
    state = "idle"          # idle | calibrating | play
    calib_start = 0.0
    melody_mode = False
    melody_idx = 0
    message = "Tekan C untuk kalibrasi"
    last_hand = time.time()
    prev = time.time()
    fps = 0.0

    while True:
        frame = tracker.read()
        if frame is None:
            print("Kamera tidak dapat dibaca.")
            break

        features, lm = tracker.detect(frame)
        now = time.time()
        fps = 0.9 * fps + 0.1 * (1.0 / max(now - prev, 1e-6))
        prev = now

        if lm is not None:
            tracker.draw(frame, lm)
            last_hand = now
        elif now - last_hand > NO_HAND_RESET_S:
            temporal.reset()
            if state == "play":
                put_text(frame, "Tangan tidak terdeteksi", 90, (0, 0, 255))

        levels = np.zeros(5)
        if state == "calibrating":
            calibrator.add(features)
            remaining = CALIB_SECONDS - (now - calib_start)
            put_text(frame, f"Kalibrasi: buka-tutup tangan berulang kali ({max(remaining, 0):.0f} dtk)", 60, (0, 255, 255))
            if remaining <= 0:
                calib = calibrator.finalize()
                if calib is None:
                    message = calibrator.message + " Tekan C."
                    state = "idle"
                else:
                    message = "Kalibrasi berhasil. Silakan bermain."
                    stats.rom = calib.rom()
                    temporal.reset()
                    state = "play"
        elif state == "play" and calib is not None and features is not None:
            levels, pressed = temporal.update(calib.bend_levels(features))
            for f in pressed:
                player.play(f)
                stats.presses[f] += 1
                if melody_mode:
                    if f == MELODY[melody_idx]:
                        stats.melody_ok += 1
                        melody_idx = (melody_idx + 1) % len(MELODY)
                    else:
                        stats.melody_bad += 1

        put_text(frame, f"FPS: {fps:.0f}  Volume: {player.volume:.1f}", 25)
        if state != "calibrating":
            put_text(frame, message, 60, (0, 255, 255))
        if state == "play" and melody_mode:
            put_text(frame, f"Melodi: tekan jari {MELODY[melody_idx] + 1} ({FINGER_NAMES[MELODY[melody_idx]]})", 95, (0, 255, 0))
            put_text(frame, f"Benar {stats.melody_ok} / Salah {stats.melody_bad}", 125)
        if state == "play":
            draw_bars(frame, levels)

        cv2.imshow("Instrumen Musik Udara", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            calibrator.reset()
            calib_start = time.time()
            state = "calibrating"
        elif key == ord("m") and state == "play":
            melody_mode = not melody_mode
            melody_idx = 0
            message = "Mode melodi " + ("aktif" if melody_mode else "mati")
        elif key in (ord("+"), ord("=")):
            player.change_volume(0.1)
        elif key == ord("-"):
            player.change_volume(-0.1)

    tracker.release()
    cv2.destroyAllWindows()
    if stats.presses.sum() > 0:
        print("Ringkasan sesi disimpan di:", stats.save())


if __name__ == "__main__":
    main()
