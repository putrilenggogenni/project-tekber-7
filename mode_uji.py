"""Mode uji untuk mengumpulkan data evaluasi. Penanggung jawab: Anggota 4.

Aplikasi meminta relawan menggerakkan jari tertentu (urutan acak). Setiap percobaan
dicatat ke CSV: jari target, jari yang memicu nada, jumlah pemicu, waktu respon, FPS.
Tidak ada video yang disimpan.

Jalankan: python mode_uji.py [--skenario normal] [--arah auto|tutup|buka]
                             [--fitur jarak|sudut] [--kamera 0]
Tombol: SPASI = mulai kalibrasi, Q = berhenti (data yang sudah terekam tetap tersimpan).

Perubahan v3:
- Kalibrasi dua fase dan syarat "istirahat" dihitung dari posisi istirahat relawan
  sendiri (tangan kaku yang tertutup tidak lagi membuat layar macet).
- Hanya jari yang aktif setelah kalibrasi yang diuji.
- Label skenario (normal, kaku_sedang, kaku_kecil, genggam, dst.), arah, dan metode
  fitur dicatat agar tiap kondisi bisa dievaluasi terpisah. Nilai default batas waktu
  (5 dtk) dan jendela (0,6 dtk) sama dengan laporan agar hasilnya tetap sebanding;
  ubah dengan --timeout dan --jendela hanya untuk kondisi uji yang berbeda dan
  laporkan sebagai kondisi terpisah.
- Reset filter 1 detik saat tangan hilang (sama dengan main.py).
"""

import argparse
import csv
import os
import random
import time

import cv2
import numpy as np

from audio import FINGER_NAMES, TonePlayer
from calibration import Calibrator, CalibrationSession
from hand_features import FEATURE_METHODS, HandTracker
from hud import draw_hud
from main import MIN_RANGE, isolation_index
from temporal import TemporalFilter

REST_SECONDS = 4
RANGE_SECONDS = 8
REST_MIN_S = 0.8        # jeda minimal antar percobaan
NO_HAND_RESET_S = 1.0
OUT_DIR = "hasil_uji"
HEADER = ["relawan", "trial", "jari_target", "jari_terdeteksi", "benar",
          "jumlah_pemicu", "waktu_respon_s", "fps",
          "skenario", "arah", "fitur", "frame_terdeteksi_persen", "isolasi", "timeout_s"]
# Kode jari: 0 jempol, 1 telunjuk, 2 tengah, 3 manis, 4 kelingking, -1 tidak terdeteksi


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Pengumpulan data uji")
    p.add_argument("--skenario", default="normal",
                   help="label kondisi uji, mis. normal, kaku_sedang, kaku_kecil, genggam")
    p.add_argument("--arah", choices=["auto", "tutup", "buka"], default="auto")
    p.add_argument("--fitur", choices=FEATURE_METHODS, default="jarak")
    p.add_argument("--kamera", type=int, default=0)
    p.add_argument("--ulang", type=int, default=20, help="percobaan per jari")
    p.add_argument("--timeout", type=float, default=5.0, help="batas waktu menunggu pemicu (dtk)")
    p.add_argument("--jendela", type=float, default=0.6, help="jendela pemicu tambahan (dtk)")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    relawan = input("Kode relawan (misalnya R1): ").strip() or "R0"
    os.makedirs(OUT_DIR, exist_ok=True)

    tracker = HandTracker(camera_index=args.kamera, method=args.fitur)
    if not tracker.is_open():
        print("Kamera tidak dapat dibuka. Periksa kamera atau coba --kamera 1.")
        tracker.release()
        return
    player = TonePlayer()
    calibrator = Calibrator(min_range=MIN_RANGE[args.fitur])
    session = CalibrationSession(calibrator, REST_SECONDS, RANGE_SECONDS)
    temporal = TemporalFilter()
    calib, direction, active = None, "tutup", np.ones(5, dtype=bool)
    trials, trial_i = [], 0
    fh, writer, out_path = None, None, None

    state = "wait_calib"   # wait_calib | calibrating | rest | prompt | collect | done
    t_state = t_prompt = t_first = time.time()
    events, first_iso = [], None
    n_frames, n_det = 0, 0
    last_hand = time.time()
    message = "Tekan SPASI untuk mulai kalibrasi"
    prev, fps = time.time(), 0.0

    def record(detected, n_trig, rt):
        target = trials[trial_i]
        pct = 100.0 * n_det / n_frames if n_frames else 0.0
        writer.writerow([relawan, trial_i + 1, target, detected, int(detected == target),
                         n_trig, round(rt, 3), round(fps, 1), args.skenario, direction,
                         args.fitur, round(pct, 1),
                         round(first_iso, 3) if first_iso is not None else "",
                         args.timeout])
        fh.flush()

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
        if state in ("prompt", "collect"):
            n_frames += 1
            n_det += int(lm is not None)

        signal, pressed = np.zeros(5), []
        if calib is not None and features is not None and state in ("rest", "prompt", "collect"):
            signal, pressed = temporal.update(calib.effort(features, direction), active)

        if state == "calibrating":
            phase, remaining, result = session.update(features, now)
            message = CalibrationSession.prompt(phase, remaining)
            if phase == "failed":
                message = calibrator.message + " Tekan SPASI."
                state = "wait_calib"
            elif phase == "done":
                calib = result
                direction = args.arah if args.arah != "auto" else calib.auto_direction()
                active = calib.active_mask(direction)
                if not active.any():
                    message = f"Arah {direction} tidak tersedia. Coba --arah lain. Tekan SPASI."
                    calib, state = None, "wait_calib"
                else:
                    ids = [i for i in range(5) if active[i]]
                    trials = [f for f in ids for _ in range(args.ulang)]
                    random.shuffle(trials)
                    trial_i = 0
                    name = f"uji_{relawan}_{args.skenario}_{direction}_{args.fitur}.csv"
                    out_path = os.path.join(OUT_DIR, name)
                    if fh:
                        fh.close()
                    fh = open(out_path, "w", newline="", encoding="utf-8")
                    writer = csv.writer(fh)
                    writer.writerow(HEADER)
                    temporal.reset()
                    state, t_state = "rest", now
                    print(f"Jari yang diuji: {[FINGER_NAMES[i] for i in ids]}; arah {direction}")
        elif state == "rest":
            message = "Kembali ke posisi istirahat, tunggu instruksi"
            ready = features is not None and not np.any(signal[active] >= temporal.off_th)
            if now - t_state > REST_MIN_S and ready:
                if trial_i >= len(trials):
                    state = "done"
                else:
                    state, t_prompt = "prompt", now
                    n_frames = n_det = 0
        elif state == "prompt":
            target = trials[trial_i]
            message = f"[{trial_i + 1}/{len(trials)}] Gerakkan: {FINGER_NAMES[target].upper()}"
            if pressed:
                events = list(pressed)
                first_iso = isolation_index(signal, active, events[0])
                for f in pressed:
                    player.play(f)
                state, t_first = "collect", now
            elif now - t_prompt > args.timeout:
                first_iso = None
                record(-1, 0, args.timeout)
                trial_i += 1
                state, t_state = "rest", now
        elif state == "collect":
            message = "..."
            for f in pressed:
                events.append(f)
                player.play(f)
            if now - t_first > args.jendela:
                record(events[0], len(events), t_first - t_prompt)
                trial_i += 1
                events = []
                state, t_state = "rest", now

        if state == "done":
            message = "Selesai. Terima kasih! Tekan Q."

        draw_hud(frame, [(message, (0, 255, 255)),
                         (f"FPS: {fps:.0f}  Skenario: {args.skenario}  Arah: {direction}",
                          (255, 255, 255))])
        cv2.imshow("Mode Uji", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32 and state == "wait_calib":
            session.start(time.time())
            state = "calibrating"

    tracker.release()
    cv2.destroyAllWindows()
    if fh:
        fh.close()
        print("Data uji disimpan di:", out_path)


if __name__ == "__main__":
    main()