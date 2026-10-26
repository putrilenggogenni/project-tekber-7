"""Mengambil gambar tiap tahap prapemrosesan dari SATU frame webcam, untuk slide.

Cara pakai (taruh file ini di folder yang sama dengan hand_features.py):
    python ambil_gambar_prapemrosesan.py          # kamera nomor 0
    python ambil_gambar_prapemrosesan.py 1        # kamera nomor 1

Tekan SPASI untuk mengambil gambar, Q untuk keluar.
Hasil ada di folder gambar_prapemrosesan/.
"""

import os
import sys

import cv2
import mediapipe as mp
import numpy as np

from hand_features import compute_features

OUT_DIR = "gambar_prapemrosesan"
FINGER_NAMES = ["Jempol", "Telunjuk", "Tengah", "Manis", "Kelingking"]
FONT = cv2.FONT_HERSHEY_SIMPLEX


def put_label(img, text, org, scale=0.7, color=(255, 255, 255)):
    """Teks dengan kotak latar gelap supaya terbaca di gambar apa pun."""
    (w, h), base = cv2.getTextSize(text, FONT, scale, 2)
    x, y = org
    cv2.rectangle(img, (x - 6, y - h - 6), (x + w + 6, y + base + 6), (0, 0, 0), -1)
    cv2.putText(img, text, (x, y), FONT, scale, color, 2, cv2.LINE_AA)


def susun_ringkasan(images, titles, cell_w=400):
    """Menyusun gambar-gambar menjadi satu baris berjudul (untuk ditempel ke slide)."""
    cells = []
    for img, title in zip(images, titles):
        h, w = img.shape[:2]
        cell_h = int(h * cell_w / w)
        cell = cv2.resize(img, (cell_w, cell_h))
        put_label(cell, title, (10, 30), 0.6)
        cells.append(cell)
    max_h = max(c.shape[0] for c in cells)
    padded = []
    for c in cells:
        pad = max_h - c.shape[0]
        padded.append(cv2.copyMakeBorder(c, 0, pad, 0, 0, cv2.BORDER_CONSTANT, value=(30, 30, 30)))
        padded.append(np.full((max_h, 6, 3), 30, dtype=np.uint8))  # garis pemisah
    return np.hstack(padded[:-1])


def buka_kamera(index):
    if os.name == "nt":
        return cv2.VideoCapture(index, cv2.CAP_DSHOW)
    return cv2.VideoCapture(index)


def main():
    index = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    cap = buka_kamera(index)
    if not cap.isOpened():
        print("Kamera tidak dapat dibuka. Tutup aplikasi lain yang memakai kamera,")
        print("atau coba nomor kamera lain: python ambil_gambar_prapemrosesan.py 1")
        return

    hands = mp.solutions.hands.Hands(
        static_image_mode=True, max_num_hands=1, min_detection_confidence=0.6)
    drawer = mp.solutions.drawing_utils
    connections = mp.solutions.hands.HAND_CONNECTIONS
    os.makedirs(OUT_DIR, exist_ok=True)
    print("Hadapkan telapak tangan terbuka ke kamera, lalu tekan SPASI. Q untuk keluar.")

    while True:
        ok, raw = cap.read()
        if not ok:
            print("Kamera tidak dapat dibaca.")
            break

        # Pratinjau dicerminkan agar nyaman dilihat; gambar yang disimpan tetap dari frame mentah.
        preview = cv2.flip(raw, 1)
        put_label(preview, "SPASI: ambil gambar   Q: keluar", (10, 30), 0.6)
        cv2.imshow("Pengambil gambar prapemrosesan", preview)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key != ord(" "):
            continue

        # Tahap 1: frame mentah (BGR, belum dicerminkan)
        tahap1 = raw.copy()
        # Tahap 2: dicerminkan
        tahap2 = cv2.flip(raw, 1)
        # Tahap 3: salinan RGB untuk MediaPipe
        tahap3 = cv2.cvtColor(tahap2, cv2.COLOR_BGR2RGB)
        tahap3.flags.writeable = False
        result = hands.process(tahap3)
        if not result.multi_hand_landmarks:
            print("Tangan tidak terdeteksi. Coba lagi dengan tangan lebih jelas.")
            continue

        lm = result.multi_hand_landmarks[0]
        h, w = tahap2.shape[:2]

        # Tahap 4: landmark digambar di atas frame yang dicerminkan
        tahap4 = tahap2.copy()
        drawer.draw_landmarks(tahap4, lm, connections)

        # Tahap 5: lima nilai fitur tekukan (jarak ujung jari / ukuran telapak)
        feats = compute_features(lm, w, h)
        tahap5 = tahap4.copy()
        if feats is not None:
            for i, (name, f) in enumerate(zip(FINGER_NAMES, feats)):
                put_label(tahap5, f"{name}: {f:.2f}", (10, 35 + 36 * i), 0.7, (0, 255, 255))

        files = {
            "1_mentah.png": tahap1,
            "2_cermin.png": tahap2,
            "3_rgb.png": np.ascontiguousarray(tahap3),  # tersimpan dengan warna tertukar, ini normal
            "4_landmark.png": tahap4,
            "5_fitur.png": tahap5,
        }
        for name, img in files.items():
            cv2.imwrite(os.path.join(OUT_DIR, name), img)

        ringkasan = susun_ringkasan(
            [tahap1, tahap2, tahap3, tahap4, tahap5],
            ["1. Frame mentah (BGR)", "2. Dicerminkan", "3. Salinan RGB",
             "4. 21 landmark", "5. Fitur tekukan"])
        cv2.imwrite(os.path.join(OUT_DIR, "ringkasan_5_tahap.png"), ringkasan)
        print(f"Tersimpan di folder {OUT_DIR}/ (5 gambar + ringkasan_5_tahap.png)")

    cap.release()
    hands.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
