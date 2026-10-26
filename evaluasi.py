"""Evaluasi sistem dari data uji."""

import csv
import glob
import os
import re

import numpy as np

from audio import FINGER_NAMES

IN_DIR = "hasil_uji"
DEFAULTS = {"skenario": "normal", "arah": "tutup", "fitur": "jarak"}


def load_rows():
    rows = []
    for path in sorted(glob.glob(os.path.join(IN_DIR, "uji_*.csv"))):
        with open(path, newline="", encoding="utf-8") as f:
            rows.extend(csv.DictReader(f))
    return rows


def group_key(r):
    return tuple((r.get(k) or DEFAULTS[k]) for k in ("skenario", "arah", "fitur"))


def confusion_matrix(rows):
    """Baris = jari target, kolom = jari terdeteksi (kolom terakhir = tidak terdeteksi)."""
    cm = np.zeros((5, 6), dtype=int)
    for r in rows:
        target, detected = int(r["jari_target"]), int(r["jari_terdeteksi"])
        cm[target, detected if detected >= 0 else 5] += 1
    return cm


def _num(r, key):
    try:
        return float(r[key])
    except (KeyError, TypeError, ValueError):
        return None


def evaluate_group(rows):
    cm = confusion_matrix(rows)
    total = int(cm.sum())
    per_finger = []
    for i in range(5):
        col, row = cm[:, i].sum(), cm[i].sum()
        precision = cm[i, i] / col if col else None
        recall = cm[i, i] / row if row else None
        per_finger.append((FINGER_NAMES[i], precision, recall, int(row)))

    sessions = {}
    for r in rows:
        s = sessions.setdefault(r["relawan"], 0)
        sessions[r["relawan"]] = s + int(r["jumlah_pemicu"]) - int(r["benar"])
    rt = [float(r["waktu_respon_s"]) for r in rows if int(r["jari_terdeteksi"]) >= 0]
    fps = [float(r["fps"]) for r in rows]
    det = [v for v in (_num(r, "frame_terdeteksi_persen") for r in rows) if v is not None]
    iso = [v for v in (_num(r, "isolasi") for r in rows) if v is not None]
    return {
        "cm": cm, "total": total,
        "accuracy": 100.0 * np.trace(cm[:, :5]) / total if total else 0.0,
        "per_finger": per_finger,
        "false_per_session": float(np.mean(list(sessions.values()))),
        "sessions": len(sessions),
        "rt": float(np.mean(rt)) if rt else None,
        "fps": float(np.mean(fps)) if fps else None,
        "det": float(np.mean(det)) if det else None,
        "iso": float(np.mean(iso)) if iso else None,
    }


def fmt(v, spec=".2f"):
    return "n/a" if v is None else format(v, spec)


def print_group(key, res):
    print(f"\n=== Kondisi: skenario={key[0]}  arah={key[1]}  fitur={key[2]} ===")
    print("Confusion matrix (baris = target, kolom = terdeteksi + tidak terdeteksi)")
    print("            " + "  ".join(f"{n[:5]:>5}" for n in FINGER_NAMES) + "  Tidak")
    for i, name in enumerate(FINGER_NAMES):
        print(f"{name:<11} " + "  ".join(f"{v:>5}" for v in res["cm"][i]))
    print("\nJari          Precision  Recall  Percobaan")
    for name, p, r, n in res["per_finger"]:
        print(f"{name:<12} {fmt(p):>9} {fmt(r):>7} {n:>9}")
    print(f"\nAkurasi: {res['accuracy']:.1f}% dari {res['total']} percobaan")
    print(f"Nada keliru per sesi: rata-rata {res['false_per_session']:.1f} "
          f"(jumlah sesi: {res['sessions']})")
    print(f"Waktu respon rata-rata: {fmt(res['rt'])} dtk | FPS rata-rata: {fmt(res['fps'], '.1f')}")
    print(f"Frame tangan terdeteksi: {fmt(res['det'], '.1f')}% | "
          f"indeks isolasi rata-rata: {fmt(res['iso'])} (1 = jari lain tidak ikut bergerak)")


def save_matrix_png(cm, path, title):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib belum terpasang, gambar dilewati.")
        return False
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(6))
    ax.set_xticklabels([n[:5] for n in FINGER_NAMES] + ["Tidak"])
    ax.set_yticks(range(5))
    ax.set_yticklabels(FINGER_NAMES)
    ax.set_xlabel("Terdeteksi")
    ax.set_ylabel("Target")
    ax.set_title(title, fontsize=9)
    for i in range(5):
        for j in range(6):
            ax.text(j, i, cm[i, j], ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
    return True


def summarize_still():
    paths = sorted(glob.glob(os.path.join(IN_DIR, "diam_*.csv")))
    if not paths:
        return []
    print("\n=== Uji kondisi diam ===")
    out = []
    for p in paths:
        with open(p, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                print(f"{r['relawan']} {r['skenario']} {r['arah']} {r['fitur']}: "
                      f"{r['pemicu_total']} pemicu palsu dalam {r['durasi_s']} dtk "
                      f"({r['pemicu_per_menit']} per menit)")
                out.append(r)
    return out


def main():
    rows = load_rows()
    if not rows:
        print("Belum ada data di", IN_DIR)
        return

    groups = {}
    for r in rows:
        groups.setdefault(group_key(r), []).append(r)

    out = os.path.join(IN_DIR, "ringkasan_evaluasi.csv")
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        for key in sorted(groups):
            res = evaluate_group(groups[key])
            print_group(key, res)
            w.writerow(["skenario", "arah", "fitur"])
            w.writerow(list(key))
            w.writerow(["jari", "precision", "recall", "percobaan"])
            for name, p, r_, n in res["per_finger"]:
                w.writerow([name, "" if p is None else round(p, 3),
                            "" if r_ is None else round(r_, 3), n])
            w.writerow(["akurasi_persen", round(res["accuracy"], 1)])
            w.writerow(["nada_keliru_per_sesi", round(res["false_per_session"], 2)])
            w.writerow(["waktu_respon_s", "" if res["rt"] is None else round(res["rt"], 2)])
            w.writerow(["fps_rata2", "" if res["fps"] is None else round(res["fps"], 1)])
            w.writerow(["frame_terdeteksi_persen", "" if res["det"] is None else round(res["det"], 1)])
            w.writerow(["indeks_isolasi", "" if res["iso"] is None else round(res["iso"], 3)])
            w.writerow([])
            slug = re.sub(r"[^a-zA-Z0-9]+", "_", "_".join(key))
            png = os.path.join(IN_DIR, f"confusion_matrix_{slug}.png")
            if save_matrix_png(res["cm"], png, " / ".join(key)):
                print("Gambar confusion matrix:", png)
                if len(groups) == 1:  # satu kondisi saja: nama berkas sama seperti versi lama
                    save_matrix_png(res["cm"], os.path.join(IN_DIR, "confusion_matrix.png"),
                                    " / ".join(key))
        for r in summarize_still():
            w.writerow(["diam", r["relawan"], r["skenario"], r["arah"], r["fitur"],
                        r["pemicu_total"], r["pemicu_per_menit"]])
    print("\nRingkasan disimpan di:", out)


if __name__ == "__main__":
    main()