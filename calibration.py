"""Kalibrasi per pengguna.

Pengguna membuka dan menutup tangan berulang kali. Untuk tiap jari dicatat
nilai terendah (tertekuk) dan tertinggi (terbuka) memakai persentil agar
tahan terhadap noise. Nilai tekukan kemudian dinyatakan 0 (terbuka) sampai
1 (tertekuk penuh) relatif terhadap rentang pribadi pengguna.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Calibration:
    lows: np.ndarray   # nilai fitur saat jari tertekuk
    highs: np.ndarray  # nilai fitur saat jari terbuka

    def bend_levels(self, features):
        """Tekukan 0 (terbuka) .. 1 (tertekuk) untuk 5 jari."""
        span = np.maximum(self.highs - self.lows, 1e-6)
        return np.clip((self.highs - features) / span, 0.0, 1.0)

    def rom(self):
        """Estimasi rentang gerak relatif tiap jari (dinormalisasi ukuran telapak)."""
        return self.highs - self.lows


class Calibrator:
    def __init__(self, min_samples=30, min_range=0.08, low_pct=5, high_pct=95):
        self.min_samples = min_samples
        self.min_range = min_range
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.message = ""
        self.samples = []

    def reset(self):
        self.samples = []
        self.message = ""

    def add(self, features):
        if features is not None:
            self.samples.append(np.asarray(features, dtype=float))

    def finalize(self):
        """Mengembalikan Calibration, atau None jika data belum memadai."""
        if len(self.samples) < self.min_samples:
            self.message = "Tangan kurang terdeteksi. Ulangi kalibrasi."
            return None
        data = np.vstack(self.samples)
        lows = np.percentile(data, self.low_pct, axis=0)
        highs = np.percentile(data, self.high_pct, axis=0)
        if np.any(highs - lows < self.min_range):
            self.message = "Gerakan kurang lebar. Buka-tutup tangan lebih jelas."
            return None
        return Calibration(lows=lows, highs=highs)
