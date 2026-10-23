# Instrumen Musik Udara untuk Rehabilitasi Motorik Jari Pascastroke

Implementasi Computer Vision untuk rehabilitasi motorik jari pasien pascastroke
melalui instrumen musik udara menggunakan OpenCV dan MediaPipe.

Gerakan jari dideteksi lewat webcam laptop dan diubah menjadi nada. Sistem juga
mencatat ringkasan sesi latihan (repetisi, estimasi rentang gerak, ketepatan nada).
Sistem ini alat bantu latihan, bukan alat medis. Tidak ada video yang disimpan.

## Cara menjalankan

1. Gunakan Python 3.10 - 3.12 (samakan versinya antar anggota).
2. Buat dan aktifkan virtual environment:
   - Windows: `python -m venv venv` lalu `venv\Scripts\activate`
   - Mac/Linux: `python -m venv venv` lalu `source venv/bin/activate`
3. Install dependensi: `pip install -r requirements.txt`
4. Jalankan: `python main.py`

## Struktur file dan penanggung jawab

| File | Isi | Penanggung jawab |
|---|---|---|
| `hand_features.py` | Webcam, MediaPipe Hands, fitur tekukan jari | Anggota 1 |
| `calibration.py` | Kalibrasi buka-tutup tangan, ambang per jari | Anggota 2 |
| `temporal.py` | Smoothing (moving average) dan hysteresis | Anggota 2 |
| `main.py` | Aplikasi utama, nada, tampilan, ringkasan sesi (CSV) | Anggota 3 |
| `evaluasi.py` | Pengolahan data uji, confusion matrix, FPS | Anggota 4 |

## Dataset

Tidak memakai dataset publik. Data uji berupa rekaman sesi relawan sehat
(dengan persetujuan), tidak disimpan sebagai video. Hanya data numerik.

## Anggota kelompok

- Anggota 1 (NRP)
- Anggota 2 (NRP)
- Anggota 3 (NRP)
- Anggota 4 (NRP)

## Tautan tambahan

- Video demo dan data besar: (isi tautan Google Drive)
