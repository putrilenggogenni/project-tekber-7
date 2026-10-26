"""Uji kondisi diam: tangan rileks dan tidak digerakkan, hitung pemicu palsu.

Jalankan: python uji_diam.py [--durasi 60] [--skenario normal] [--arah auto|tutup|buka]
                             [--fitur jarak|sudut] [--kamera 0]
Hasil: hasil_uji/diam_<relawan>_<skenario>_<arah>_<fitur>.csv
Target di rencana laporan: pemicu palsu <= 1 per menit.
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np

from audio import FINGER_NAMES
from calibration import Calibrator, CalibrationSession
from hand_features import FEATURE_METHODS, HandTracker
from hud import draw_hud
from main import MIN_RANGE
from temporal import TemporalFilter

REST_SECONDS = 4
RANGE_SECONDS = 8
OUT_DIR = "hasil_uji"
NO_HAND_RESET_S = 1.0
HEADER = (["relawan", "skenario", "arah", "fitur", "durasi_s", "pemicu_total",
           "pemicu_per_menit", "frame_terdeteksi_persen"]
          + [f"pemicu_{n.lower()}" for n in FINGER_NAMES])


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Uji kondisi diam")
    p.add_argument("--durasi", type=float, default=60.0)
    p.add_argument("--skenario", default="normal")
    p.add_argument("--arah", choices=["auto", "tutup", "buka"], default="auto")
    p.add_argument("--fitur", choices=FEATURE_METHODS, default="jarak")
    p.add_argument("--kamera", type=int, default=0)
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
    calibrator = Calibrator(min_range=MIN_RANGE[args.fitur])
    session = CalibrationSession(calibrator, REST_SECONDS, RANGE_SECONDS)
    temporal = TemporalFilter()
    calib, direction, active = None, "tutup", np.ones(5, dtype=bool)
    counts = np.zeros(5, dtype=int)
    state, t_start = "wait", time.time()   # wait | calibrating | measuring | done
    n_frames = n_det = 0
    last_hand = time.time()
    message = "Tekan SPASI untuk kalibrasi"
    out_path = None

    while True:
        frame = tracker.read()
        if frame is None:
            print("Kamera tidak dapat dibaca.")
            break
        features, lm = tracker.detect(frame)
        now = time.time()
        if lm is not None:
            tracker.draw(frame, lm)
            last_hand = now
        elif now - last_hand > NO_HAND_RESET_S:
            temporal.reset()

        if state == "calibrating":
            phase, remaining, result = session.update(features, now)
            message = CalibrationSession.prompt(phase, remaining)
            if phase == "failed":
                message, state = calibrator.message + " Tekan SPASI.", "wait"
            elif phase == "done":
                calib = result
                direction = args.arah if args.arah != "auto" else calib.auto_direction()
                active = calib.active_mask(direction)
                if not active.any():
                    message, state = f"Arah {direction} tidak tersedia. Tekan SPASI.", "wait"
                else:
                    temporal.reset()
                    state, t_start = "measuring", now
        elif state == "measuring":
            n_frames += 1
            n_det += int(lm is not None)
            if features is not None:
                _, pressed = temporal.update(calib.effort(features, direction), active)
                for f in pressed:
                    counts[f] += 1
            left = args.durasi - (now - t_start)
            message = (f"Rilekskan tangan, JANGAN digerakkan ({max(left, 0):.0f} dtk). "
                       f"Pemicu: {int(counts.sum())}")
            if left <= 0:
                state = "done"
                dur = now - t_start
                total = int(counts.sum())
                name = f"diam_{relawan}_{args.skenario}_{direction}_{args.fitur}.csv"
                out_path = os.path.join(OUT_DIR, name)
                with open(out_path, "w", newline="", encoding="utf-8") as fh:
                    w = csv.writer(fh)
                    w.writerow(HEADER)
                    w.writerow([relawan, args.skenario, direction, args.fitur, round(dur, 1),
                                total, round(total / (dur / 60.0), 2),
                                round(100.0 * n_det / max(n_frames, 1), 1)]
                               + [int(c) for c in counts])
                print(f"Pemicu palsu: {total} dalam {dur:.0f} dtk. Disimpan di: {out_path}")
        elif state == "done":
            message = "Selesai. Tekan Q."

        draw_hud(frame, [(message, (0, 255, 255))])
        cv2.imshow("Uji Kondisi Diam", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32 and state == "wait":
            session.start(time.time())
            state = "calibrating"

    tracker.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
