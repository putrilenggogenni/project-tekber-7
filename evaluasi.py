"""Evaluasi sistem dari data uji.

Membaca semua file hasil_uji/uji_*.csv lalu menghitung:
confusion matrix per jari, precision, recall, akurasi, nada keliru per sesi,
rata-rata waktu respon dan FPS. Jalankan: python evaluasi.py
"""

import csv
import glob
import os

import numpy as np

from audio import FINGER_NAMES

IN_DIR = "hasil_uji"


def load_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(IN_DIR, "uji_*.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def confusion_matrix(rows):
    """Baris = jari target, kolom = jari terdeteksi (kolom terakhir = tidak terdeteksi)."""
    cm = np.zeros((5, 6), dtype=int)
    for r in rows:
        target, detected = int(r["jari_target"]), int(r["jari_terdeteksi"])
        cm[target, detected if detected >= 0 else 5] += 1
    return cm


def main():
    rows = load_rows()
    if not rows:
        print("Belum ada data di", IN_DIR)
        return

    cm = confusion_matrix(rows)
    total = cm.sum()
    accuracy = np.trace(cm[:, :5]) / total

    print("Confusion matrix (baris = target, kolom = terdeteksi + tidak terdeteksi)")
    print("            " + "  ".join(f"{n[:5]:>5}" for n in FINGER_NAMES) + "  Tidak")
    for i, name in enumerate(FINGER_NAMES):
        print(f"{name:<11} " + "  ".join(f"{v:>5}" for v in cm[i]))

    summary = [["jari", "precision", "recall"]]
    print("\nJari          Precision  Recall")
    for i, name in enumerate(FINGER_NAMES):
        col, row = cm[:, i].sum(), cm[i].sum()
        precision = cm[i, i] / col if col else 0.0
        recall = cm[i, i] / row if row else 0.0
        summary.append([name, round(precision, 3), round(recall, 3)])
        print(f"{name:<12} {precision:>9.2f} {recall:>7.2f}")

    print(f"\nAkurasi keseluruhan: {accuracy * 100:.1f}% dari {total} percobaan")

    # Nada keliru = pemicu yang tidak sesuai target (pemicu tambahan atau jari salah)
    sessions = {}
    for r in rows:
        s = sessions.setdefault(r["relawan"], {"false": 0})
        s["false"] += int(r["jumlah_pemicu"]) - int(r["benar"])
    false_per_session = [v["false"] for v in sessions.values()]
    mean_false = float(np.mean(false_per_session))
    print(f"Nada keliru per sesi: rata-rata {mean_false:.1f} (jumlah sesi: {len(sessions)})")

    rt = [float(r["waktu_respon_s"]) for r in rows if int(r["jari_terdeteksi"]) >= 0]
    fps = [float(r["fps"]) for r in rows]
    print(f"Rata-rata waktu respon: {np.mean(rt):.2f} dtk" if rt else "Tidak ada waktu respon.")
    print(f"Rata-rata FPS: {np.mean(fps):.1f}")

    out = os.path.join(IN_DIR, "ringkasan_evaluasi.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerows(summary)
        w.writerow([])
        w.writerow(["akurasi_persen", round(accuracy * 100, 1)])
        w.writerow(["nada_keliru_per_sesi", round(mean_false, 2)])
        w.writerow(["fps_rata2", round(float(np.mean(fps)), 1)])
    print("Ringkasan disimpan di:", out)

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(6, 5))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(6))
        ax.set_xticklabels([n[:5] for n in FINGER_NAMES] + ["Tidak"])
        ax.set_yticks(range(5))
        ax.set_yticklabels(FINGER_NAMES)
        ax.set_xlabel("Terdeteksi")
        ax.set_ylabel("Target")
        for i in range(5):
            for j in range(6):
                ax.text(j, i, cm[i, j], ha="center", va="center")
        fig.tight_layout()
        fig.savefig(os.path.join(IN_DIR, "confusion_matrix.png"), dpi=150)
        print("Gambar confusion matrix: ", os.path.join(IN_DIR, "confusion_matrix.png"))
    except ImportError:
        print("matplotlib belum terpasang, gambar dilewati.")


if __name__ == "__main__":
    main()
