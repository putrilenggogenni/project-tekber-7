# Instrumen Musik Udara untuk Rehabilitasi Motorik Jari Pascastroke

Implementasi Computer Vision untuk rehabilitasi motorik jari pasien pascastroke
melalui instrumen musik udara menggunakan OpenCV dan MediaPipe.

Gerakan jari dideteksi lewat webcam laptop dan diubah menjadi nada. Sistem juga
mencatat ringkasan sesi latihan (repetisi, estimasi rentang gerak, ketepatan melodi).
Sistem ini alat bantu latihan, bukan alat medis. Tidak ada video yang disimpan.

## Cara menjalankan

1. Gunakan Python 3.10 - 3.12 (samakan versinya antar anggota). Versi MediaPipe dikunci di requirements.txt, jangan di-upgrade.
2. Buat dan aktifkan virtual environment:
   - Windows: `python -m venv venv` lalu `venv\Scripts\activate`
   - Mac/Linux: `python -m venv venv` lalu `source venv/bin/activate`
3. Install dependensi: `pip install -r requirements.txt`
4. Jalankan aplikasi: `python main.py`

Tombol di aplikasi: `C` kalibrasi (buka-tutup tangan selama 8 detik), `M` mode melodi,
`+` / `-` volume, `Q` keluar dan simpan ringkasan sesi ke `hasil_sesi/ringkasan_sesi.csv`.

## Pengujian dan evaluasi

1. `python mode_uji.py` : relawan mengikuti instruksi jari (5 jari x 20 percobaan),
   data tersimpan di `hasil_uji/uji_<kode relawan>.csv`.
2. `python evaluasi.py` : menghitung confusion matrix, precision, recall,
   nada keliru per sesi, waktu respon, dan FPS dari semua file di `hasil_uji/`.

Kode jari di CSV: 0 jempol, 1 telunjuk, 2 tengah, 3 manis, 4 kelingking, -1 tidak terdeteksi.

## Struktur file dan penanggung jawab

| File | Isi | Penanggung jawab |
|---|---|---|
| `hand_features.py` | Webcam, MediaPipe Hands, fitur tekukan jari | Anggota 1 |
| `calibration.py` | Kalibrasi buka-tutup tangan, ambang per jari | Anggota 2 |
| `temporal.py` | Smoothing (moving average) dan hysteresis | Anggota 2 |
| `audio.py` | Pembuatan dan pemutaran nada | Anggota 3 |
| `main.py` | Aplikasi utama, tampilan, mode melodi, ringkasan sesi | Anggota 3 |
| `mode_uji.py` | Pengumpulan data uji dari relawan | Anggota 4 |
| `evaluasi.py` | Confusion matrix, precision, recall, nada keliru, FPS | Anggota 4 |

## Dataset

Tidak memakai dataset publik. Data uji berupa log numerik sesi relawan sehat
(dengan persetujuan, identitas diganti kode). Tidak ada video yang disimpan.

## Anggota kelompok

- Anggota 1 (NRP)
- Anggota 2 (NRP)
- Anggota 3 (NRP)
- Anggota 4 (NRP)

## Tautan tambahan

- Video demo dan data besar: (isi tautan Google Drive)
