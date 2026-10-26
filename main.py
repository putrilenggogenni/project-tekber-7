"""Aplikasi utama: instrumen musik udara (v3). Penanggung jawab: Anggota 3.

Jalankan: python main.py [--kamera 0] [--fitur jarak|sudut]

Tombol:
  C = kalibrasi 2 fase (istirahat 4 dtk, lalu gerak 8 dtk)
  B = arah latihan TUTUP/BUKA (otomatis dipilih setelah kalibrasi)
  H = tugas seluruh tangan (satu akord per ulangan)
  M = mode melodi        D = ganti level melodi     T = melodi ketat/toleran
  A = tandai sesi dibantu/mandiri                   P = jeda
  [ / ] = sensitivitas   + / - = volume             Q = keluar, simpan, ringkasan
"""

import argparse
import csv
import os
import time
from collections import deque
from datetime import datetime

import cv2
import numpy as np

from audio import FINGER_NAMES, TonePlayer
from calibration import Calibrator, CalibrationSession
from hand_features import FEATURE_METHODS, HandTracker
from hud import GREEN, RED, WHITE, YELLOW, draw_hud, put_text
from temporal import TemporalFilter

REST_SECONDS = 4
RANGE_SECONDS = 8
MIN_RANGE = {"jarak": 0.05, "sudut": 8.0}   # rentang gerak minimal per metode fitur
OUT_DIR = "hasil_sesi"
NO_HAND_RESET_S = 1.0
FLASH_S = 0.35
FLASH_OK = (0, 200, 0)
FLASH_BAD = (0, 0, 255)
BREAK_S = 300          # pengingat istirahat setiap 5 menit waktu latihan aktif
BREAK_NOTICE_S = 30    # lama pengingat tampil
SMALL_HAND_RATIO = 0.12  # ukuran telapak / lebar frame di bawah ini = terlalu jauh

# Level melodi. "batas": None = tanpa batas waktu; "adaptif" = mengikuti kecepatan pengguna.
LEVELS = [
    {"kode": "mudah", "nama": "Mudah (3 jari)",
     "melodi": [0, 1, 2, 1, 0, 1, 2, 1], "batas": None},
    {"kode": "sedang", "nama": "Sedang (5 jari)",
     "melodi": [0, 1, 2, 3, 4, 3, 2, 1], "batas": None},
    {"kode": "waktu", "nama": "Tantangan waktu (adaptif)",
     "melodi": [0, 1, 2, 3, 4, 3, 2, 1], "batas": "adaptif"},
]


def time_limit(level, note_times):
    """Batas waktu per nada (detik) atau None. Mode adaptif: 3x median waktu benar
    terakhir, dibatasi 8-30 detik; awalnya 12 detik."""
    batas = level["batas"]
    if batas is None:
        return None
    if batas == "adaptif":
        if len(note_times) < 3:
            return 12.0
        return float(np.clip(3.0 * np.median(note_times), 8.0, 30.0))
    return float(batas)


def isolation_index(signal, active, finger):
    """1.0 = hanya jari target yang bergerak; makin kecil = jari lain ikut bergerak.
    None jika tidak ada jari aktif lain sebagai pembanding."""
    others = [signal[i] for i in range(len(signal)) if active[i] and i != finger]
    if not others:
        return None
    return float(signal[finger] / (signal[finger] + float(np.mean(others)) + 1e-6))


def draw_bars(frame, levels, target=None, active=None):
    h, w = frame.shape[:2]
    bar_w, bar_h = 30, 100
    x0, y0 = w - 200, h - 130
    for i, lv in enumerate(levels):
        x = x0 + i * 38
        is_active = True if active is None else bool(active[i])
        edge = (90, 90, 90) if is_active else (60, 60, 160)
        cv2.rectangle(frame, (x, y0), (x + bar_w, y0 + bar_h), edge, 1)
        if is_active:
            fill = int(bar_h * float(lv))
            cv2.rectangle(frame, (x, y0 + bar_h - fill), (x + bar_w, y0 + bar_h), (0, 200, 0), -1)
        else:
            cv2.line(frame, (x, y0), (x + bar_w, y0 + bar_h), edge, 1)
        if target is not None and i == target:
            cv2.rectangle(frame, (x - 3, y0 - 3), (x + bar_w + 3, y0 + bar_h + 3), (0, 255, 255), 3)
        cv2.putText(frame, str(i + 1), (x + 9, y0 + bar_h + 18),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)


def draw_flash(frame, color, thickness=10):
    h, w = frame.shape[:2]
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, thickness)


class SessionStats:
    def __init__(self, method="jarak"):
        self.method = method
        self.start = datetime.now()
        self.end = None
        self.presses = np.zeros(5, dtype=int)
        self.rom = np.zeros(5)         # rentang gerak dari kalibrasi (satuan fitur)
        self.max_effort = np.zeros(5)  # usaha maksimum (0..1) yang dicapai selama sesi
        self.active = np.ones(5, dtype=bool)
        self.melody = np.zeros((len(LEVELS), 3), dtype=int)  # [benar, salah, terlewat]
        self.hand_reps = 0
        self.iso = []                  # indeks isolasi tiap pemicu jari
        self.frames = 0                # frame saat latihan aktif
        self.frames_det = 0
        self.frames_two = 0
        self.frames_small = 0
        self.assist_seen = set()
        self.direction_seen = set()

    def finish(self):
        self.end = datetime.now()

    @property
    def duration(self):
        end = self.end or datetime.now()
        return (end - self.start).total_seconds()

    @staticmethod
    def accuracy(row):
        total = int(np.sum(row))
        return 100.0 * float(row[0]) / total if total else None

    def assist_label(self):
        if len(self.assist_seen) > 1:
            return "campuran"
        return next(iter(self.assist_seen), "mandiri")

    def iso_mean(self):
        return float(np.mean(self.iso)) if self.iso else None

    def pct(self, n):
        return 100.0 * n / self.frames if self.frames else 0.0

    def save(self):
        os.makedirs(OUT_DIR, exist_ok=True)
        path = os.path.join(OUT_DIR, "ringkasan_sesi.csv")
        header = (["waktu_mulai", "durasi_detik"]
                  + [f"repetisi_{n.lower()}" for n in FINGER_NAMES]
                  + [f"rom_{n.lower()}" for n in FINGER_NAMES]
                  + ["melodi_benar", "melodi_salah", "melodi_terlewat",
                     "akurasi_melodi_persen", "level_melodi"]
                  + [f"gerak_maks_{n.lower()}" for n in FINGER_NAMES]
                  + ["mode_bantuan", "arah_picu", "jari_aktif",
                     "ulangan_tangan", "indeks_isolasi", "frame_terdeteksi_persen",
                     "frame_dua_tangan_persen", "frame_tangan_kecil_persen", "fitur"])
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
        iso = self.iso_mean()
        row = ([self.start.strftime("%Y-%m-%d %H:%M:%S"), round(self.duration, 1)]
               + [int(x) for x in self.presses]
               + [round(float(x), 3) for x in self.rom]
               + [int(tot[0]), int(tot[1]), int(tot[2]),
                  round(acc, 1) if acc is not None else "", "+".join(used)]
               + [round(float(x), 3) for x in self.max_effort]
               + [self.assist_label(), "+".join(sorted(self.direction_seen)),
                  "".join("1" if a else "0" for a in self.active),
                  self.hand_reps, round(iso, 3) if iso is not None else "",
                  round(self.pct(self.frames_det), 1), round(self.pct(self.frames_two), 1),
                  round(self.pct(self.frames_small), 1), self.method])
        with open(path, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(header)
            writer.writerow(row)
        return path


def render_summary(stats, saved_path):
    canvas = np.full((620, 820, 3), 30, dtype=np.uint8)
    font = cv2.FONT_HERSHEY_SIMPLEX

    def txt(text, x, y, color=WHITE, scale=0.6, th=1):
        cv2.putText(canvas, text, (x, y), font, scale, color, th, cv2.LINE_AA)

    mm, ss = divmod(int(stats.duration), 60)
    txt("RINGKASAN SESI LATIHAN", 20, 40, YELLOW, 0.85, 2)
    txt(f"Durasi: {mm:02d}:{ss:02d}   Total repetisi jari: {int(stats.presses.sum())}"
        f"   Sesi: {stats.assist_label()}", 20, 78)

    txt("Jari", 20, 118, YELLOW)
    txt("Repetisi", 170, 118, YELLOW)
    txt("Rentang kalibrasi", 290, 118, YELLOW)
    txt("Usaha maks sesi (0-1)", 500, 118, YELLOW)
    for i, name in enumerate(FINGER_NAMES):
        y = 150 + 30 * i
        txt(name + ("" if stats.active[i] else " (nonaktif)"), 20, y)
        txt(str(int(stats.presses[i])), 170, y)
        rom = float(stats.rom[i])
        txt(f"{rom:.2f}" if rom > 0 else "-", 290, y)
        mx = float(stats.max_effort[i])
        txt(f"{mx:.2f}" if mx > 0 else "-", 500, y)

    y = 338
    iso = stats.iso_mean()
    txt(f"Ulangan seluruh tangan: {stats.hand_reps}     "
        f"Indeks isolasi jari: {iso:.2f} (1 = jari lain tidak ikut bergerak)"
        if iso is not None else f"Ulangan seluruh tangan: {stats.hand_reps}     Indeks isolasi jari: -",
        20, y, WHITE, 0.55)
    y += 40
    txt("Ketepatan melodi per level", 20, y, YELLOW)
    for i, lv in enumerate(LEVELS):
        y += 30
        row = stats.melody[i]
        acc = stats.accuracy(row)
        if acc is None:
            txt(f"{lv['nama']}: belum dimainkan", 20, y)
        else:
            txt(f"{lv['nama']}: benar {row[0]}, salah {row[1]}, "
                f"terlewat {row[2]} ({acc:.0f}%)", 20, y)
    tot_acc = stats.accuracy(stats.melody.sum(axis=0))
    y += 40
    txt("Akurasi melodi keseluruhan: " + (f"{tot_acc:.0f}%" if tot_acc is not None else "-"),
        20, y, GREEN)
    y += 36
    txt(f"Tangan terdeteksi {stats.pct(stats.frames_det):.0f}% frame | dua tangan "
        f"{stats.pct(stats.frames_two):.0f}% | tangan terlalu kecil "
        f"{stats.pct(stats.frames_small):.0f}%", 20, y, (200, 200, 200), 0.5)
    txt(f"Disimpan di: {saved_path}", 20, 570, (200, 200, 200), 0.5)
    txt("Tekan tombol apa saja untuk menutup.", 20, 598, (200, 200, 200), 0.5)
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


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Instrumen musik udara untuk rehabilitasi jari")
    p.add_argument("--kamera", type=int, default=0, help="nomor kamera (default 0)")
    p.add_argument("--fitur", choices=FEATURE_METHODS, default="jarak",
                   help="metode fitur: jarak (laporan) atau sudut")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    tracker = HandTracker(camera_index=args.kamera, method=args.fitur)
    if not tracker.is_open():
        print("Kamera tidak dapat dibuka. Periksa kamera atau coba --kamera 1.")
        tracker.release()
        return
    player = TonePlayer()
    calibrator = Calibrator(min_range=MIN_RANGE[args.fitur])
    session = CalibrationSession(calibrator, REST_SECONDS, RANGE_SECONDS)
    temporal = TemporalFilter()
    hand_filter = TemporalFilter(n_fingers=1)
    stats = SessionStats(args.fitur)

    calib = None
    direction = "tutup"
    active = np.ones(5, dtype=bool)
    state = "idle"          # idle | calibrating | play
    paused = False
    hand_mode = False       # tugas seluruh tangan
    strict = False
    assisted = False
    on_th = temporal.on_th
    melody_mode = False
    level_idx = 0
    melody_idx = 0
    note_times = deque(maxlen=8)
    note_start = time.time()
    flash_color, flash_until = None, 0.0
    message = "Tekan C untuk kalibrasi"
    last_hand = time.time()
    prev = time.time()
    fps = 0.0
    active_time = 0.0
    next_break = BREAK_S
    break_until = 0.0

    def reset_filters():
        temporal.reset()
        hand_filter.reset()

    while True:
        frame = tracker.read()
        if frame is None:
            print("Kamera tidak dapat dibaca.")
            break

        features, lm = tracker.detect(frame)
        now = time.time()
        dt = now - prev
        fps = 0.9 * fps + 0.1 * (1.0 / max(dt, 1e-6))
        prev = now
        playing = state == "play" and not paused
        if playing:
            active_time += dt
            stats.frames += 1
            if lm is not None:
                stats.frames_det += 1
                if tracker.n_hands >= 2:
                    stats.frames_two += 1
                if tracker.palm_ratio < SMALL_HAND_RATIO:
                    stats.frames_small += 1
            if active_time >= next_break:
                next_break += BREAK_S
                break_until = now + BREAK_NOTICE_S

        level = LEVELS[level_idx]
        melody = [f for f in level["melodi"] if active[f]]
        if melody_mode and not melody:
            melody_mode = False
            message = "Melodi tidak bisa dipakai: jari melodi nonaktif."
        melody_idx = melody_idx % len(melody) if melody else 0

        warn = []
        if lm is not None:
            tracker.draw(frame, lm)
            last_hand = now
            if state == "play":
                if tracker.n_hands >= 2:
                    warn.append(("Dua tangan terlihat: jangan menutupi jari pasien", YELLOW))
                if tracker.palm_ratio < SMALL_HAND_RATIO:
                    warn.append(("Tangan terlalu kecil di layar: dekatkan ke kamera", YELLOW))
        elif now - last_hand > NO_HAND_RESET_S:
            reset_filters()
            if state == "play":
                warn.append(("Tangan tidak terdeteksi", RED))
        if lm is None:
            note_start = now

        signal = np.zeros(5)
        calib_text = None
        if state == "calibrating":
            phase, remaining, result = session.update(features, now)
            calib_text = CalibrationSession.prompt(phase, remaining)
            if phase == "failed":
                message = calibrator.message + " Tekan C."
                state = "idle"
            elif phase == "done":
                calib = result
                direction = calib.auto_direction()
                active = calib.active_mask(direction)
                stats.active = active.copy()
                stats.rom = calib.rom()
                message = (f"Kalibrasi berhasil. Arah otomatis: {direction.upper()}. "
                           f"{calibrator.message}")
                reset_filters()
                note_start = now
                melody_idx = 0
                state = "play"
        elif state == "play" and paused:
            note_start = now
            warn.append(("JEDA - tekan P untuk lanjut", YELLOW))
        elif state == "play" and calib is not None and features is not None:
            eff = calib.effort(features, direction)
            signal, pressed_f = temporal.update(eff, active)
            stats.max_effort = np.maximum(stats.max_effort, signal)
            stats.direction_seen.add(direction)
            if hand_mode:
                level_hand = float(np.mean(eff[active])) if active.any() else 0.0
                _, hand_pressed = hand_filter.update(np.array([level_hand]))
                if hand_pressed:
                    stats.hand_reps += 1
                    stats.assist_seen.add("dibantu" if assisted else "mandiri")
                    player.play_chord()
                    flash_color, flash_until = FLASH_OK, now + FLASH_S
                pressed = []
            else:
                pressed = pressed_f
            for f in pressed:
                stats.presses[f] += 1
                stats.assist_seen.add("dibantu" if assisted else "mandiri")
                iso = isolation_index(signal, active, f)
                if iso is not None:
                    stats.iso.append(iso)
                if not melody_mode:
                    player.play(f)
                    continue
                if f == melody[melody_idx]:
                    player.play(f)
                    stats.melody[level_idx, 0] += 1
                    note_times.append(now - note_start)
                    melody_idx = (melody_idx + 1) % len(melody)
                    note_start = now
                    flash_color, flash_until = FLASH_OK, now + FLASH_S
                elif strict:
                    player.play_wrong()
                    stats.melody[level_idx, 1] += 1
                    flash_color, flash_until = FLASH_BAD, now + FLASH_S
                # mode toleran: jari lain yang ikut bergerak diabaikan tanpa hukuman

        limit = time_limit(level, note_times)
        if state == "play" and not paused and melody_mode and melody and limit is not None:
            if now - note_start > limit:
                stats.melody[level_idx, 2] += 1
                melody_idx = (melody_idx + 1) % len(melody)
                note_start = now
                flash_color, flash_until = FLASH_BAD, now + FLASH_S

        if now < break_until:
            warn.append(("Sudah 5 menit berlatih. Waktunya istirahat sebentar.", YELLOW))

        target = melody[melody_idx] if (state == "play" and melody_mode and melody) else None

        hud = [(f"FPS: {fps:.0f}   Volume: {player.volume:.1f}", WHITE),
               (calib_text if calib_text else message, YELLOW)]
        if state == "play":
            mode_name = "SELURUH TANGAN" if hand_mode else ("MELODI" if melody_mode else "BEBAS")
            hud.append((f"Arah: {direction.upper()}  Mode: {mode_name}  Sens: {on_th:.2f}  "
                        f"Sesi: {'DIBANTU' if assisted else 'MANDIRI'}", WHITE))
            if hand_mode:
                hud.append((f"Ulangan seluruh tangan: {stats.hand_reps}", GREEN))
        hud.extend(warn)
        if target is not None:
            ok, bad, miss = (int(v) for v in stats.melody[level_idx])
            hud.append((f"Level: {level['nama']}", WHITE))
            hud.append((f"Melodi: gerakkan jari {target + 1} ({FINGER_NAMES[target]})", GREEN))
            hud.append((f"Benar {ok} / Salah {bad} / Terlewat {miss}", WHITE))
            if limit is not None:
                hud.append((f"Sisa waktu: {max(limit - (now - note_start), 0.0):.0f} dtk", YELLOW))
        draw_hud(frame, hud)

        if state == "play":
            draw_bars(frame, signal, target, active)
        if flash_color is not None and now < flash_until:
            draw_flash(frame, flash_color)
        if state != "calibrating":
            put_text(frame, "C:kalib B:arah H:tangan M:melodi D:level T:ketat A:dibantu P:jeda Q:keluar",
                     frame.shape[0] - 15, WHITE, 0.4)

        cv2.imshow("Instrumen Musik Udara", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord("c"):
            session.start(time.time())
            paused = False
            state = "calibrating"
        elif key == ord("p") and state == "play":
            paused = not paused
            reset_filters()
            message = "Jeda" if paused else "Lanjut"
        elif key == ord("m") and state == "play":
            melody_mode = not melody_mode
            hand_mode = False if melody_mode else hand_mode
            melody_idx = 0
            note_start = time.time()
            message = "Mode melodi " + ("aktif" if melody_mode else "mati")
        elif key == ord("h") and state == "play":
            hand_mode = not hand_mode
            melody_mode = False if hand_mode else melody_mode
            reset_filters()
            message = ("Tugas seluruh tangan aktif: gerakkan semua jari bersama" if hand_mode
                       else "Tugas seluruh tangan mati")
        elif key == ord("d") and state == "play":
            level_idx = (level_idx + 1) % len(LEVELS)
            melody_idx = 0
            note_times.clear()
            note_start = time.time()
            message = f"Level melodi: {LEVELS[level_idx]['nama']}"
        elif key == ord("b") and state == "play" and calib is not None:
            other = "buka" if direction == "tutup" else "tutup"
            mask = calib.active_mask(other)
            if mask.any():
                direction, active = other, mask
                stats.active = active.copy()
                reset_filters()
                message = ("Nada saat jari MEMBUKA dari posisi istirahat" if direction == "buka"
                           else "Nada saat jari MENUTUP dari posisi istirahat")
            else:
                message = f"Arah {other.upper()} tidak tersedia pada kalibrasi ini."
        elif key == ord("t"):
            strict = not strict
            message = "Melodi ketat (jari salah dihitung)" if strict else "Melodi toleran (jari lain diabaikan)"
        elif key == ord("a"):
            assisted = not assisted
            message = "Sesi ditandai DIBANTU" if assisted else "Sesi ditandai MANDIRI"
        elif key == ord("]"):
            on_th = min(on_th + 0.05, 0.8)
            temporal.set_threshold(on_th)
            hand_filter.set_threshold(on_th)
            message = f"Sensitivitas: {on_th:.2f} (makin kecil makin mudah memicu)"
        elif key == ord("["):
            on_th = max(on_th - 0.05, 0.3)
            temporal.set_threshold(on_th)
            hand_filter.set_threshold(on_th)
            message = f"Sensitivitas: {on_th:.2f} (makin kecil makin mudah memicu)"
        elif key in (ord("+"), ord("=")):
            player.change_volume(0.1)
        elif key == ord("-"):
            player.change_volume(-0.1)

    tracker.release()
    cv2.destroyAllWindows()
    stats.finish()
    if stats.presses.sum() > 0 or stats.hand_reps > 0:
        path = stats.save()
        print("Ringkasan sesi disimpan di:", path)
        show_summary(stats, path)


if __name__ == "__main__":
    main()