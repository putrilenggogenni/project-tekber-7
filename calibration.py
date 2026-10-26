"""Kalibrasi per pengguna. Penanggung jawab: Anggota 2.

Perubahan v3 (untuk tangan kaku yang cenderung tertutup):
- Kalibrasi dua fase: (1) POSISI ISTIRAHAT: tangan dibiarkan rileks dan diam,
  (2) RENTANG GERAK: pengguna menggerakkan jari semampunya.
- Sinyal usaha dihitung RELATIF terhadap posisi istirahat, ke dua arah:
  "tutup" (menjauh dari istirahat ke arah menekuk) dan "buka" (ke arah meluruskan).
- Noise saat istirahat diukur dan dipakai sebagai ZONA MATI: gerakan sekecil noise
  tidak dihitung sebagai usaha.
- Validitas per jari dan per arah: jari dengan rentang terlalu kecil (atau tidak
  jelas lebih besar dari noise) dinonaktifkan, bukan menggagalkan seluruh kalibrasi.
"""

from dataclasses import dataclass

import numpy as np

_NAMES = ["Jempol", "Telunjuk", "Tengah", "Manis", "Kelingking"]


@dataclass
class Calibration:
    lows: np.ndarray            # nilai fitur saat jari paling tertekuk (persentil rendah)
    highs: np.ndarray           # nilai fitur saat jari paling terbuka (persentil tinggi)
    rest: np.ndarray = None     # posisi istirahat (median); None = tidak direkam
    noise: np.ndarray = None    # simpangan noise saat istirahat
    valid_close: np.ndarray = None
    valid_open: np.ndarray = None
    dead_k: float = 3.0         # zona mati = dead_k x noise

    def __post_init__(self):
        n = len(self.lows)
        if self.noise is None:
            self.noise = np.zeros(n)
        if self.valid_close is None:
            self.valid_close = np.ones(n, dtype=bool)
        if self.valid_open is None:
            self.valid_open = np.ones(n, dtype=bool)

    @property
    def valid(self):
        """Kompatibilitas: jari yang valid di setidaknya satu arah."""
        return self.valid_close | self.valid_open

    def dead_zone(self):
        return self.dead_k * np.asarray(self.noise, dtype=float)

    def _reference(self, direction):
        """Posisi acuan 0%. Tanpa rekaman istirahat: tangan sehat (terbuka untuk
        'tutup', tertutup untuk 'buka'), sama seperti perilaku versi sebelumnya."""
        if self.rest is not None:
            return np.asarray(self.rest, dtype=float)
        return self.highs if direction == "tutup" else self.lows

    def span(self, direction):
        ref = self._reference(direction)
        return (ref - self.lows) if direction == "tutup" else (self.highs - ref)

    def effort(self, features, direction):
        """Usaha 0..1 per jari: seberapa jauh jari bergerak dari posisi istirahat ke
        arah yang dilatih, setelah dikurangi zona mati."""
        f = np.asarray(features, dtype=float)
        ref = self._reference(direction)
        disp = (ref - f) if direction == "tutup" else (f - ref)
        dead = self.dead_zone()
        usable = np.maximum(self.span(direction) - dead, 1e-6)
        return np.clip((disp - dead) / usable, 0.0, 1.0)

    def active_mask(self, direction):
        return self.valid_close.copy() if direction == "tutup" else self.valid_open.copy()

    def auto_direction(self):
        """Arah dengan jari valid lebih banyak. Seri -> 'tutup'."""
        return "buka" if self.valid_open.sum() > self.valid_close.sum() else "tutup"

    def bend_levels(self, features):
        """Tekukan 0 (terbuka) .. 1 (tertekuk) relatif terhadap rentang gerak.
        Dipertahankan agar kode lama tetap berjalan."""
        span = np.maximum(self.highs - self.lows, 1e-6)
        return np.clip((self.highs - np.asarray(features, dtype=float)) / span, 0.0, 1.0)

    def rom(self):
        """Estimasi rentang gerak relatif tiap jari (satuan fitur)."""
        return self.highs - self.lows


class Calibrator:
    def __init__(self, min_samples=30, min_rest_samples=20, min_range=0.05,
                 min_snr=6.0, dead_k=3.0, low_pct=5, high_pct=95):
        """min_range memakai satuan fitur (0.05 untuk 'jarak', sekitar 8 untuk 'sudut').
        min_snr: rentang gerak minimal sebagai kelipatan noise saat istirahat."""
        self.min_samples = min_samples
        self.min_rest_samples = min_rest_samples
        self.min_range = min_range
        self.min_snr = min_snr
        self.dead_k = dead_k
        self.low_pct = low_pct
        self.high_pct = high_pct
        self.message = ""
        self.samples = []
        self.rest_samples = []

    def reset(self):
        self.samples = []
        self.rest_samples = []
        self.message = ""

    def add_rest(self, features):
        if features is not None:
            self.rest_samples.append(np.asarray(features, dtype=float))

    def add(self, features):
        """Menambah sampel fase rentang gerak."""
        if features is not None:
            self.samples.append(np.asarray(features, dtype=float))

    def finalize(self, require_rest=False):
        """Mengembalikan Calibration, atau None jika data belum memadai."""
        if len(self.samples) < self.min_samples:
            self.message = "Tangan kurang terdeteksi. Ulangi kalibrasi."
            return None
        data = np.vstack(self.samples)
        lows = np.percentile(data, self.low_pct, axis=0)
        highs = np.percentile(data, self.high_pct, axis=0)

        rest, noise = None, np.zeros(len(lows))
        if len(self.rest_samples) >= self.min_rest_samples:
            rs = np.vstack(self.rest_samples)
            rest = np.median(rs, axis=0)
            noise = 1.4826 * np.median(np.abs(rs - rest), axis=0)  # simpangan robust
        elif require_rest:
            self.message = "Posisi istirahat belum terekam. Ulangi kalibrasi."
            return None

        calib = Calibration(lows=lows, highs=highs, rest=rest, noise=noise,
                            dead_k=self.dead_k)
        floor = np.maximum(self.min_range, self.min_snr * noise)
        calib.valid_close = calib.span("tutup") >= floor
        calib.valid_open = calib.span("buka") >= floor
        if not np.any(calib.valid):
            self.message = ("Belum ada jari yang bergerak cukup jelas dibanding noise. "
                            "Gerakkan jari lebih jelas atau perbaiki pencahayaan.")
            return None

        parts = []
        for label, mask in (("tutup", calib.valid_close), ("buka", calib.valid_open)):
            off = [_NAMES[i] for i in range(len(mask)) if not mask[i]]
            if len(off) == len(mask):
                parts.append(f"Arah {label} tidak tersedia.")
            elif off:
                parts.append(f"Arah {label}: jari nonaktif {', '.join(off)}.")
        self.message = " ".join(parts)
        return calib


class CalibrationSession:
    """Alur kalibrasi dua fase yang dipakai bersama oleh main.py dan skrip uji."""

    def __init__(self, calibrator, rest_s=4.0, range_s=8.0):
        self.calibrator = calibrator
        self.rest_s = rest_s
        self.range_s = range_s
        self.phase = "idle"   # idle | rest | range | done | failed
        self.t0 = 0.0

    def start(self, now):
        self.calibrator.reset()
        self.phase, self.t0 = "rest", now

    def update(self, features, now):
        """Mengembalikan (fase, sisa detik, Calibration atau None)."""
        if self.phase == "rest":
            self.calibrator.add_rest(features)
            remaining = self.rest_s - (now - self.t0)
            if remaining <= 0:
                self.phase, self.t0, remaining = "range", now, self.range_s
            return self.phase, remaining, None
        if self.phase == "range":
            self.calibrator.add(features)
            remaining = self.range_s - (now - self.t0)
            if remaining <= 0:
                calib = self.calibrator.finalize(require_rest=True)
                self.phase = "done" if calib is not None else "failed"
                return self.phase, 0.0, calib
            return "range", remaining, None
        return self.phase, 0.0, None

    @staticmethod
    def prompt(phase, remaining):
        if phase == "rest":
            return f"Kalibrasi 1/2: istirahatkan tangan, jangan digerakkan ({max(remaining, 0):.0f} dtk)"
        if phase == "range":
            return f"Kalibrasi 2/2: gerakkan jari semampumu ({max(remaining, 0):.0f} dtk)"
        return ""