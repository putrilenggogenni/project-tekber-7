"""Pembuat dan pemutar nada. Penanggung jawab: Anggota 3.

Perubahan v3: play_chord() untuk tugas "seluruh tangan" (jempol, tengah, kelingking
berbunyi bersamaan sebagai satu akord C-E-G).
"""

import numpy as np
import pygame

FINGER_NAMES = ["Jempol", "Telunjuk", "Tengah", "Manis", "Kelingking"]
FINGER_FREQS = [261.63, 293.66, 329.63, 349.23, 392.00]  # C4 D4 E4 F4 G4
WRONG_FREQ = 150.0  # nada rendah dan kasar sebagai penanda jari keliru
CHORD_FINGERS = (0, 2, 4)  # C4 E4 G4


class TonePlayer:
    def __init__(self, volume=0.5, duration=0.4, sample_rate=44100):
        pygame.mixer.init(frequency=sample_rate, size=-16, channels=1)
        self.volume = volume
        self.sounds = [self._make_tone(f, duration, sample_rate) for f in FINGER_FREQS]
        self.wrong_sound = self._make_wrong(0.25, sample_rate)

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

    @staticmethod
    def _make_wrong(duration, sr):
        """Bunyi 'buzz' pendek (gelombang kotak rendah) untuk jari yang keliru."""
        t = np.linspace(0, duration, int(sr * duration), endpoint=False)
        wave = 0.6 * np.sign(np.sin(2 * np.pi * WRONG_FREQ * t)) + 0.4 * np.sin(2 * np.pi * (WRONG_FREQ + 5) * t)
        env = np.ones_like(wave)
        attack, release = int(0.005 * sr), int(0.06 * sr)
        env[:attack] = np.linspace(0, 1, attack)
        env[-release:] = np.linspace(1, 0, release)
        samples = (wave * env * 32767 * 0.6).astype(np.int16)
        return pygame.mixer.Sound(buffer=samples.tobytes())

    def play(self, finger_index):
        sound = self.sounds[finger_index]
        sound.set_volume(self.volume)
        sound.play()

    def play_chord(self):
        """Akord C-E-G, dipakai untuk satu ulangan gerakan seluruh tangan."""
        for i in CHORD_FINGERS:
            sound = self.sounds[i]
            sound.set_volume(self.volume * 0.6)  # lebih pelan agar tidak terlalu keras
            sound.play()

    def play_wrong(self):
        self.wrong_sound.set_volume(self.volume)
        self.wrong_sound.play()

    def change_volume(self, delta):
        self.volume = float(np.clip(self.volume + delta, 0.0, 1.0))