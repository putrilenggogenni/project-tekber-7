"""Analisis temporal: smoothing dan hysteresis."""

from collections import deque

import numpy as np


class TemporalFilter:
    def __init__(self, n_fingers=5, window=5, on_th=0.6, off_th=0.35):
        self.n = n_fingers
        self.window = window
        self.on_th = on_th    # jari dianggap "aktif" saat sinyal >= on_th
        self.off_th = off_th  # dianggap "dilepas" saat sinyal <= off_th
        self.reset()

    def set_threshold(self, on_th, gap=0.25):
        """Ubah sensitivitas. Ambang lepas selalu di bawah ambang picu (hysteresis)."""
        self.on_th = float(on_th)
        self.off_th = max(self.on_th - gap, 0.05)

    def reset(self):
        self.buffers = [deque(maxlen=self.window) for _ in range(self.n)]
        self.pressed = np.zeros(self.n, dtype=bool)
        self.first = True

    def update(self, levels, active=None):
        """Mengembalikan (sinyal terhaluskan, daftar indeks jari yang BARU memicu).

        levels : 5 nilai 0..1 (makin besar = makin dekat ke gerakan target)
        active : mask bool 5 jari; jari nonaktif tidak pernah memicu.
        """
        if active is None:
            active = np.ones(self.n, dtype=bool)
        smoothed = np.zeros(self.n)
        new_presses = []
        for i in range(self.n):
            if not active[i]:
                self.pressed[i] = False
                continue
            self.buffers[i].append(float(levels[i]))
            smoothed[i] = float(np.mean(self.buffers[i]))
            if self.first:
                # Frame pertama: hanya catat posisi awal, jangan picu nada.
                self.pressed[i] = smoothed[i] >= self.on_th
            elif not self.pressed[i] and smoothed[i] >= self.on_th:
                self.pressed[i] = True
                new_presses.append(i)
            elif self.pressed[i] and smoothed[i] <= self.off_th:
                self.pressed[i] = False
        self.first = False
        return smoothed, new_presses