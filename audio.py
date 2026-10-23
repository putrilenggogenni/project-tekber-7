"""Pembuat dan pemutar nada. Penanggung jawab: Anggota 3."""

import numpy as np
import pygame

FINGER_NAMES = ["Jempol", "Telunjuk", "Tengah", "Manis", "Kelingking"]
FINGER_FREQS = [261.63, 293.66, 329.63, 349.23, 392.00]  # C4 D4 E4 F4 G4


class TonePlayer:
    def __init__(self, volume=0.5, duration=0.4, sample_rate=44100):
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=1)
        self.volume = volume
        self.sounds = [self._make_tone(f, duration, sample_rate) for f in FINGER_FREQS]

    @staticmethod
    def _make_tone(freq, duration, sr):
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        wave = np.sin(2 * np.pi * freq * t)
        env = np.ones_like(wave)
        attack, release = int(0.01 * sr), int(0.1 * sr)
        env[:attack] = np.linspace(0, 1, attack)
        env[-release:] = np.linspace(1, 0, release)
        samples = (wave * env * 32767 * 0.8).astype(np.int16)
        return pygame.mixer.Sound(buffer=samples.tobytes())

    def play(self, finger_index):
        sound = self.sounds[finger_index]
        sound.set_volume(self.volume)
        sound.play()

    def change_volume(self, delta):
        self.volume = float(np.clip(self.volume + delta, 0.0, 1.0))
