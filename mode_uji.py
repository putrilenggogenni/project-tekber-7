"""Mode uji untuk mengumpulkan data evaluasi..

Aplikasi meminta relawan menggerakkan jari tertentu (urutan acak). Setiap percobaan
dicatat ke CSV: jari target, jari yang memicu nada, jumlah pemicu, waktu respon, FPS.
Tidak ada video yang disimpan. Jalankan: python mode_uji.py
Tombol: SPASI = mulai kalibrasi, Q = berhenti (data yang sudah terekam tetap tersimpan).
"""

import csv
import os
import random
import time

import cv2
import numpy as np

from audio import FINGER_NAMES, TonePlayer
from calibration import Calibrator
from hand_features import HandTracker
from hud import draw_hud
from temporal import TemporalFilter

REPEATS = 20            # percobaan per jari
CALIB_SECONDS = 8
PROMPT_TIMEOUT_S = 5.0  # batas waktu menunggu pemicu
COLLECT_S = 0.6         # jendela mengumpulkan pemicu tambahan setelah pemicu pertama
REST_MIN_S = 0.8        # jeda minimal antar percobaan
OUT_DIR = "hasil_uji"
HEADER = ["relawan", "trial", "jari_target", "jari_terdeteksi", "benar",
          "jumlah_pemicu", "waktu_respon_s", "fps"]
# Kode jari: 0 jempol, 1 telunjuk, 2 tengah, 3 manis, 4 kelingking, -1 tidak terdeteksi


def main():
    relawan = input("Kode relawan (misalnya R1): ").strip() or "R0"
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"uji_{relawan}.csv")

    trials = [f for f in range(5) for _ in range(REPEATS)]
    random.shuffle(trials)

    tracker = HandTracker()
    player = TonePlayer()
    calibrator = Calibrator()
    temporal = TemporalFilter()
    calib = None

    state = "wait_calib"   # wait_calib | calibrating | rest | prompt | collect | done
    trial_i = 0
    t_state = t_prompt = t_first = time.time()
    events = []
    message = "Tekan SPASI untuk mulai kalibrasi"
    prev, fps = time.time(), 0.0

    with open(out_path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(HEADER)

        def record(detected, n_trig, rt):
            target = trials[trial_i]
            writer.writerow([relawan, trial_i + 1, target, detected,
                             int(detected == target), n_trig, round(rt, 3), round(fps, 1)])
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

            levels, pressed = np.zeros(5), []
            if calib is not None and features is not None and state in ("rest", "prompt", "collect"):
                levels, pressed = temporal.update(calib.bend_levels(features))

            if state == "calibrating":
                calibrator.add(features)
                remaining = CALIB_SECONDS - (now - t_state)
                message = f"Kalibrasi: buka-tutup tangan ({max(remaining, 0):.0f} dtk)"
                if remaining <= 0:
                    calib = calibrator.finalize()
                    if calib is None:
                        message = calibrator.message + " Tekan SPASI."
                        state = "wait_calib"
                    else:
                        temporal.reset()
                        state, t_state = "rest", now
            elif state == "rest":
                message = "Buka tangan, tunggu instruksi"
                if now - t_state > REST_MIN_S and np.all(levels < temporal.off_th):
                    if trial_i >= len(trials):
                        state = "done"
                    else:
                        state, t_prompt = "prompt", now
            elif state == "prompt":
                target = trials[trial_i]
                message = f"[{trial_i + 1}/{len(trials)}] Gerakkan: {FINGER_NAMES[target].upper()}"
                if pressed:
                    events = list(pressed)
                    for f in pressed:
                        player.play(f)
                    state, t_first = "collect", now
                elif now - t_prompt > PROMPT_TIMEOUT_S:
                    record(-1, 0, PROMPT_TIMEOUT_S)
                    trial_i += 1
                    state, t_state = "rest", now
            elif state == "collect":
                message = "..."
                for f in pressed:
                    events.append(f)
                    player.play(f)
                if now - t_first > COLLECT_S:
                    record(events[0], len(events), t_first - t_prompt)
                    trial_i += 1
                    events = []
                    state, t_state = "rest", now

            if state == "done":
                message = "Selesai. Terima kasih! Tekan Q."

            draw_hud(frame, [(message, (0, 255, 255)), (f"FPS: {fps:.0f}", (255, 255, 255))])
            cv2.imshow("Mode Uji", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break
            if key == 32 and state == "wait_calib":
                calibrator.reset()
                state, t_state = "calibrating", time.time()

    tracker.release()
    cv2.destroyAllWindows()
    print("Data uji disimpan di:", out_path)


if __name__ == "__main__":
    main()