"""Cek apakah tangan dengan postur tertentu (genggam, setengah terbuka, terbuka)
terdeteksi MediaPipe, dan metode fitur mana yang memisahkan postur lebih jelas.

Jalankan: python uji_deteksi_postur.py [--detik 8] [--kamera 0]
Alur: untuk tiap postur, tahan tangan 8 detik tanpa banyak bergerak.
Keluaran (layar + hasil_uji/postur_<relawan>.csv):
- persen frame yang tangannya terdeteksi per postur
- rata-rata dan simpangan tiap jari untuk fitur "jarak" dan "sudut"
- pemisahan genggam vs terbuka = selisih rata-rata / simpangan gabungan (makin besar
  makin jelas terpisah; di bawah sekitar 2 berarti tumpang-tindih dengan noise)
Ini langkah pertama yang perlu dilakukan sebelum memperbaiki yang lain: kalau tangan
menggenggam sering tidak terdeteksi, perbaikan lain tidak banyak berguna.
"""

import argparse
import csv
import os
import time

import cv2
import numpy as np

from audio import FINGER_NAMES
from hand_features import FEATURE_METHODS, HandTracker, compute_features
from hud import draw_hud

OUT_DIR = "hasil_uji"
POSES = [("genggam", "GENGGAM tangan (seperti posisi istirahat pasien)"),
         ("setengah", "BUKA SEBAGIAN tangan"),
         ("terbuka", "BUKA tangan selebar mungkin")]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Cek deteksi postur tangan")
    p.add_argument("--detik", type=float, default=8.0)
    p.add_argument("--kamera", type=int, default=0)
    return p.parse_args(argv)


def separation(a, b):
    """|mean(a)-mean(b)| / simpangan gabungan, per jari. a,b: (n, 5)."""
    pooled = np.sqrt((np.var(a, axis=0) + np.var(b, axis=0)) / 2.0)
    return np.abs(np.mean(a, axis=0) - np.mean(b, axis=0)) / np.maximum(pooled, 1e-9)


def main(argv=None):
    args = parse_args(argv)
    relawan = input("Kode relawan (misalnya R1): ").strip() or "R0"
    os.makedirs(OUT_DIR, exist_ok=True)
    tracker = HandTracker(camera_index=args.kamera)
    if not tracker.is_open():
        print("Kamera tidak dapat dibuka. Periksa kamera atau coba --kamera 1.")
        tracker.release()
        return

    data = {name: {"frames": 0, "det": 0, **{m: [] for m in FEATURE_METHODS}} for name, _ in POSES}
    pose_i, state, t0 = 0, "ready", time.time()   # ready | hold | done
    message = "Tekan SPASI untuk mulai postur pertama"

    while True:
        frame = tracker.read()
        if frame is None:
            print("Kamera tidak dapat dibaca.")
            break
        _, lm = tracker.detect(frame)
        now = time.time()
        if lm is not None:
            tracker.draw(frame, lm)

        if state == "hold":
            name = POSES[pose_i][0]
            d = data[name]
            d["frames"] += 1
            if lm is not None:
                h, w = frame.shape[:2]
                fj, fs = (compute_features(lm, w, h, m) for m in FEATURE_METHODS)
                if fj is not None and fs is not None:
                    d["det"] += 1
                    d["jarak"].append(fj)
                    d["sudut"].append(fs)
            left = args.detik - (now - t0)
            message = f"{POSES[pose_i][1]} ({max(left, 0):.0f} dtk)"
            if left <= 0:
                pose_i += 1
                if pose_i >= len(POSES):
                    state = "done"
                else:
                    state, message = "ready", f"Tekan SPASI untuk: {POSES[pose_i][1]}"
        elif state == "done":
            message = "Selesai. Lihat hasil di terminal. Tekan Q."

        draw_hud(frame, [(message, (0, 255, 255))])
        cv2.imshow("Cek Deteksi Postur", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == 32 and state == "ready":
            state, t0 = "hold", time.time()
        if state == "done":
            break

    tracker.release()
    cv2.destroyAllWindows()
    report(data, relawan)


def report(data, relawan):
    rows = []
    print("\n=== Deteksi tangan per postur ===")
    for name, _ in POSES:
        d = data[name]
        pct = 100.0 * d["det"] / d["frames"] if d["frames"] else 0.0
        print(f"{name:<9} terdeteksi {pct:5.1f}% dari {d['frames']} frame")
        rows.append([relawan, name, "deteksi_persen", "", round(pct, 1)] + [""] * 5)
    for m in FEATURE_METHODS:
        print(f"\n=== Fitur '{m}': rata-rata ± simpangan per jari ===")
        for name, _ in POSES:
            arr = np.array(data[name][m])
            if len(arr) == 0:
                print(f"{name:<9} (tidak ada data)")
                continue
            mean, sd = arr.mean(axis=0), arr.std(axis=0)
            print(f"{name:<9} " + "  ".join(f"{FINGER_NAMES[i][:4]} {mean[i]:.2f}±{sd[i]:.2f}"
                                           for i in range(5)))
            rows.append([relawan, name, "rata2", m] + [""] + [round(float(x), 3) for x in mean])
            rows.append([relawan, name, "simpangan", m] + [""] + [round(float(x), 3) for x in sd])
        a, b = np.array(data["genggam"][m]), np.array(data["terbuka"][m])
        if len(a) > 5 and len(b) > 5:
            sep = separation(a, b)
            print("Pemisahan genggam vs terbuka: "
                  + "  ".join(f"{FINGER_NAMES[i][:4]} {sep[i]:.1f}" for i in range(5)))
            rows.append([relawan, "genggam_vs_terbuka", "pemisahan", m, ""]
                        + [round(float(x), 2) for x in sep])
    path = os.path.join(OUT_DIR, f"postur_{relawan}.csv")
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["relawan", "postur", "ukuran", "fitur", "nilai"]
                   + [n.lower() for n in FINGER_NAMES])
        w.writerows(rows)
    print("\nDisimpan di:", path)


if __name__ == "__main__":
    main()
