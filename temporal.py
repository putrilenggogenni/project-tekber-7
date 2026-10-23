"""Analisis temporal: smoothing dan hysteresis. Penanggung jawab: Anggota 2."""

from collections import deque

import numpy as np


class TemporalFilter:
    def __init__(self, n_fingers=5, window=5, on_th=0.6, off_th=0.35):
        self.n = n_fingers
        self.window = window
        self.on_th = on_th    # jari dianggap "ditekan" saat tekukan >= on_th
        self.off_th = off_th  # dianggap "dilepas" saat tekukan <= off_th
        self.reset()

    def reset(self):
        self.buffers = [deque(maxlen=self.window) for _ in range(self.n)]
        self.pressed = np.zeros(self.n, dtype=bool)

    def update(self, bend_levels):
        """Mengembalikan (tekukan terhaluskan, daftar indeks jari yang BARU ditekan)."""
        smoothed = np.zeros(self.n)
        new_presses = []
        for i in range(self.n):
            self.buffers[i].append(float(bend_levels[i]))
            smoothed[i] = float(np.mean(self.buffers[i]))
            if not self.pressed[i] and smoothed[i] >= self.on_th:
                self.pressed[i] = True
                new_presses.append(i)
            elif self.pressed[i] and smoothed[i] <= self.off_th:
                self.pressed[i] = False
        return smoothed, new_presses
