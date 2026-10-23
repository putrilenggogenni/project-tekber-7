"""Aplikasi utama: instrumen musik udara. Penanggung jawab: Anggota 3.

Tombol: C = kalibrasi, M = mode melodi, D = ganti level melodi,
+ / - = volume, Q = keluar, simpan, dan tampilkan ringkasan sesi.
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
from hud import GREEN, RED, WHITE, YELLOW, draw_hud, put_text
from temporal import TemporalFilter

CALIB_SECONDS = 8
OUT_DIR = "hasil_sesi"
NO_HAND_RESET_S = 1.0
FLASH_S = 0.35
FLASH_OK = (0, 200, 0)
FLASH_BAD = (0, 0, 255)

# Level melodi: jumlah jari (isi melodi) dan kecepatan (batas waktu per nada, detik)
LEVELS = [
    {"kode": "mudah", "nama": "Mudah (3 jari)",
     "melodi": [0, 1, 2, 1, 0, 1, 2, 1], "batas": None},
    {"kode": "sedang", "nama": "Sedang (5 jari, 6 dtk)",
     "melodi": [0, 1, 2, 3, 4, 3, 2, 1], "batas": 6.0},
    {"kode": "sulit", "nama": "Sulit (5 jari, 3 dtk)",
     "melodi": [0, 1, 2, 3, 4, 3, 2, 1], "batas": 3.0},
]


def draw_bars(frame, levels, target=None):
    h, w = frame.shape[:2]
    bar_w, bar_h = 30, 100
    x0, y0 = w - 200, h - 130
    for i, lv in enumerate(levels):
        x = x0 + i * 38
        cv2.rectangle(frame, (x, y0), (x + bar_w, y0 + bar_h), (90, 90, 90), 1)
        fill = int(bar_h * float(lv))
        cv2.rectangle(frame, (x, y0 + bar_h - fill), (x + bar_w, y0 + bar_h), (0, 200, 0), -1)
        if target is not None and i == target:  # tandai jari target melodi
            cv2.rectangle(frame, (x - 3, y0 - 3), (x + bar_w + 3, y0 + bar_h + 3), (0, 255, 255), 3)
        cv2.putText(frame, str(i + 1), (x + 9, y0 + bar_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_flash(frame, color, thickness=10):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)


class SessionStats:
    def __init__(self):
        self.start = datetime.now()
        self.end = None
        self.presses = np.zeros(5, dtype=int)
        self.rom = np.zeros(5)
        # per level: [benar, salah, terlewat]
        self.melody = np.zeros((len(LEVELS), 3), dtype=int)

    def finish(self):
        self.end = datetime.now()

    @property
    def duration(self):
        end = self.end or datetime.now()
        return (end - self.start).total_seconds()

    @staticmethod
    def accuracy(row):
        """Akurasi (persen) dari [benar, salah, terlewat]; None jika belum ada data."""
        total = int(np.sum(row))
        return 100.0 * float(row[0]) / total if total else None

    def save(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "ringkasan_sesi.csv")
        header = (["waktu_mulai", "durasi_detik"]
                  + [f"repetisi_{n.lower()}" for n in FINGER_NAMES]
                  + [f"rom_{n.lower()}" for n in FINGER_NAMES]
                  + ["melodi_benar", "melodi_salah", "melodi_terlewat",
                     "akurasi_melodi_persen", "level_melodi"])
        # Jika file lama memakai format kolom berbeda, simpan sebagai cadangan
        if os.path.exists(path):
            with open(path, newline="", encoding="utf-8") as f:
                first_row = next(csv.reader(f), [])
            if first_row != header:
                backup = os.path.join(
                    OUT_DIR, f"ringkasan_sesi_lama_{datetime.now():%Y%m%d_%H%M%S}.csv")
                os.replace(path, backup)
        new_file = not os.path.exists(path)

        tot = self.melody.sum(axis=0)
        acc = self.accuracy(tot)
        used = [lv["kode"] for lv, r in zip(LEVELS, self.melody) if r.sum() > 0]
        row = ([self.start.strftime("%Y-%m-%d %H:%M:%S"), round(self.duration, 1)]
               + [int(x) for x in self.presses]
               + [round(float(x), 3) for x in self.rom]
               + [int(tot[0]), int(tot[1]), int(tot[2]),
                  round(acc, 1) if acc is not None else "", "+".join(used)])
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(header)
            writer.writerow(row)
        return path


def render_summary(stats, saved_path):
    """Membuat gambar layar ringkasan sesi."""
    canvas = np.full((540, 800, 3), 30, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def txt(text, x, y, color=WHITE, scale=0.6, th=1):
        cv2.putText(canvas, text, (x, y), font, scale, color, th, cv2.LINE_AA)

    mm, ss = divmod(int(stats.duration), 60)
    txt("RINGKASAN SESI LATIHAN", 20, 40, YELLOW, 0.85, 2)
    txt(f"Durasi: {mm:02d}:{ss:02d}     Total repetisi: {int(stats.presses.sum())}", 20, 78)

    txt("Jari", 20, 118, YELLOW)
    txt("Repetisi", 200, 118, YELLOW)
    txt("Estimasi rentang gerak (relatif)", 330, 118, YELLOW)
    for i, name in enumerate(FINGER_NAMES):
        y = 150 + 30 * i
        txt(name, 20, y)
        txt(str(int(stats.presses[i])), 200, y)
        rom = float(stats.rom[i])
        txt(f"{rom:.2f}" if rom > 0 else "-", 330, y)

    txt("Ketepatan melodi per level", 20, 328, YELLOW)
    for i, lv in enumerate(LEVELS):
        y = 358 + 30 * i
        row = stats.melody[i]
        acc = stats.accuracy(row)
        if acc is None:
            txt(f"{lv['nama']}: belum dimainkan", 20, y)
        else:
            txt(f"{lv['nama']}: benar {row[0]}, salah {row[1]}, "
                f"terlewat {row[2]} ({acc:.0f}%)", 20, y)
    tot_acc = stats.accuracy(stats.melody.sum(axis=0))
    txt("Akurasi melodi keseluruhan: " + (f"{tot_acc:.0f}%" if tot_acc is not None else "-"),
        20, 358 + 30 * len(LEVELS) + 10, GREEN)

    txt(f"Disimpan di: {saved_path}", 20, 490, (200, 200, 200), 0.5)
    txt("Tekan tombol apa saja untuk menutup.", 20, 518, (200, 200, 200), 0.5)
    return canvas


def show_summary(stats, saved_path):
    win = "Ringkasan Sesi"
    cv2.imshow(win, render_summary(stats, saved_path))
    while True:
        if cv2.waitKey(100) != -1:
            break
        if cv2.getWindowProperty(win, cv2.WND_PROP_VISIBLE) < 1:
            break
    cv2.destroyAllWindows()


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
    level_idx = 0
    melody_idx = 0
    note_start = time.time()
    flash_color, flash_until = None, 0.0
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

        level = LEVELS[level_idx]
        melody = level["melodi"]

        warn = None
        if lm is not None:
            tracker.draw(frame, lm)
            last_hand = now
        elif now - last_hand > NO_HAND_RESET_S:
            temporal.reset()
            if state == "play":
                warn = ("Tangan tidak terdeteksi", RED)
        if lm is None:
            note_start = now  # penghitung waktu per nada berhenti saat tangan tidak terlihat

        levels = np.zeros(5)
        calib_text = None
        if state == "calibrating":
            calibrator.add(features)
            remaining = CALIB_SECONDS - (now - calib_start)
            calib_text = f"Kalibrasi: buka-tutup tangan ({max(remaining, 0):.0f} dtk)"
            if remaining <= 0:
                calib = calibrator.finalize()
                if calib is None:
                    message = calibrator.message + " Tekan C."
                    state = "idle"
                else:
                    message = "Kalibrasi berhasil. Silakan bermain."
                    stats.rom = calib.rom()
                    temporal.reset()
                    note_start = now
                    state = "play"
        elif state == "play" and calib is not None and features is not None:
            levels, pressed = temporal.update(calib.bend_levels(features))
            for f in pressed:
                stats.presses[f] += 1
                if not melody_mode:
                    player.play(f)
                elif f == melody[melody_idx]:
                    player.play(f)
                    stats.melody[level_idx, 0] += 1
                    melody_idx = (melody_idx + 1) % len(melody)
                    note_start = now
                    flash_color, flash_until = FLASH_OK, now + FLASH_S
                else:
                    player.play_wrong()
                    stats.melody[level_idx, 1] += 1
                    flash_color, flash_until = FLASH_BAD, now + FLASH_S

        # Batas waktu per nada (level Sedang dan Sulit): nada terlewat dihitung
        if state == "play" and melody_mode and level["batas"] is not None:
            if now - note_start > level["batas"]:
                stats.melody[level_idx, 2] += 1
                melody_idx = (melody_idx + 1) % len(melody)
                note_start = now
                flash_color, flash_until = FLASH_BAD, now + FLASH_S

        target = melody[melody_idx] if (state == "play" and melody_mode) else None

        # Susunan teks dari atas ke bawah
        hud = [(f"FPS: {fps:.0f}   Volume: {player.volume:.1f}", WHITE),
               (calib_text if calib_text else message, YELLOW)]
        if warn:
            hud.append(warn)
        if target is not None:
            ok, bad, miss = (int(v) for v in stats.melody[level_idx])
            hud.append((f"Level: {level['nama']}", WHITE))
            hud.append((f"Melodi: tekan jari {target + 1} ({FINGER_NAMES[target]})", GREEN))
            hud.append((f"Benar {ok} / Salah {bad} / Terlewat {miss}", WHITE))
            if level["batas"] is not None:
                sisa = max(level["batas"] - (now - note_start), 0.0)
                hud.append((f"Sisa waktu: {sisa:.0f} dtk", YELLOW))
        draw_hud(frame, hud)

        if state == "play":
            draw_bars(frame, levels, target)
        if flash_color is not None and now < flash_until:
            draw_flash(frame, flash_color)
        if state != "calibrating":
            put_text(frame, "C: kalibrasi  M: melodi  D: level  Q: keluar",
                     frame.shape[0] - 15, WHITE, 0.5)

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
            note_start = time.time()
            message = "Mode melodi " + ("aktif" if melody_mode else "mati")
        elif key == ord("d") and state == "play":
            level_idx = (level_idx + 1) % len(LEVELS)
            melody_idx = 0
            note_start = time.time()
            message = f"Level melodi: {LEVELS[level_idx]['nama']}"
        elif key in (ord("+"), ord("=")):
            player.change_volume(0.1)
        elif key == ord("-"):
            player.change_volume(-0.1)

    tracker.release()
    cv2.destroyAllWindows()
    stats.finish()
    if stats.presses.sum() > 0:
        path = stats.save()
        print("Ringkasan sesi disimpan di:", path)
        show_summary(stats, path)


if __name__ == "__main__":
    main()