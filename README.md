# Instrumen Musik Udara untuk Rehabilitasi Motorik Jari Pascastroke

Implementasi Computer Vision untuk rehabilitasi motorik jari pasien pascastroke
melalui instrumen musik udara menggunakan OpenCV dan MediaPipe.

Gerakan jari dideteksi lewat webcam laptop dan diubah menjadi nada (C4 sampai G4,
satu nada per jari). Sistem juga mencatat ringkasan sesi latihan (repetisi,
estimasi rentang gerak, ketepatan melodi).
Sistem ini alat bantu latihan, bukan alat medis. Tidak ada video yang disimpan.

## Fitur

- Deteksi satu tangan dengan MediaPipe Hands (21 landmark) dari webcam.
- Fitur tekukan tiap jari relatif terhadap ukuran telapak, dinormalisasi lewat
  kalibrasi per pengguna (buka-tutup tangan selama 8 detik).
- Smoothing (rata-rata bergerak 5 frame) dan hysteresis (tekan pada tekukan 0,6
  ke atas, lepas pada 0,35 ke bawah) agar satu gerakan dihitung satu repetisi.
- Mode bebas (tiap jari membunyikan nadanya) dan mode melodi dengan 3 level.
- Ringkasan sesi di layar dan ekspor ke CSV.
- Mode uji dan skrip evaluasi untuk mengukur akurasi deteksi.

## Cara menjalankan

1. Gunakan Python 3.12.10 Versi MediaPipe dikunci di requirements.txt, jangan di-upgrade.
2. Buat dan aktifkan virtual environment:
   - Windows: `python -m venv venv` lalu `venv\Scripts\activate`
   - Mac/Linux: `python -m venv venv` lalu `source venv/bin/activate`
3. Install dependensi: `pip install -r requirements.txt`
4. Hubungkan webcam (aplikasi memakai kamera default, indeks 0), lalu jalankan: `python main.py`

## Cara memakai aplikasi (`main.py`)

1. Saat aplikasi terbuka, tekan `C` lalu buka dan tutup tangan berulang kali selama
   8 detik. Jika kalibrasi gagal (tangan kurang terdeteksi atau gerakan kurang lebar),
   muncul pesan di layar dan kalibrasi bisa diulang dengan `C`.
2. Setelah kalibrasi berhasil, tekuk jari untuk membunyikan nadanya (mode bebas).
3. Tekan `M` untuk mode melodi dan `D` untuk mengganti level.
4. Tekan `Q` untuk keluar.

| Tombol | Fungsi |
|---|---|
| `C` | Kalibrasi (buka-tutup tangan selama 8 detik) |
| `M` | Mode melodi hidup/mati (aktif setelah kalibrasi berhasil) |
| `D` | Ganti level melodi (aktif setelah kalibrasi berhasil) |
| `+` atau `=` / `-` | Volume naik / turun |
| `Q` | Keluar, simpan, dan tampilkan ringkasan sesi |

### Mode melodi

| Level | Jari dalam melodi | Batas waktu per nada |
|---|---|---|
| Mudah | 3 jari (jempol, telunjuk, tengah) | Tanpa batas |
| Sedang | 5 jari | 6 detik |
| Sulit | 5 jari | 3 detik |

Jari target ditandai kotak kuning pada batang tekukan. Jika jari yang ditekuk benar,
nada berbunyi, melodi maju, dan layar berkedip hijau. Jika salah, terdengar nada
rendah yang kasar, layar berkedip merah, dan melodi tidak maju. Pada level Sedang
dan Sulit, nada yang melewati batas waktu dihitung terlewat dan melodi lanjut ke
nada berikutnya.

### Ringkasan sesi

Setelah `Q`, ringkasan sesi tampil di layar (tutup dengan menekan tombol apa saja)
dan disimpan ke `hasil_sesi/ringkasan_sesi.csv`, dengan syarat ada minimal satu
repetisi pada sesi tersebut. Isinya durasi sesi, repetisi per jari, estimasi rentang
gerak relatif (diambil dari hasil kalibrasi), dan akurasi melodi per level. Jika
format kolom file lama berbeda, file lama diganti nama menjadi
`ringkasan_sesi_lama_<waktu>.csv` sebelum file baru dibuat. Folder `hasil_sesi/`
tercantum di `.gitignore` sehingga tidak ikut repositori.

## Pengujian dan evaluasi

1. `python mode_uji.py` : masukkan kode relawan di terminal (misalnya `R1`), tekan
   `SPASI` untuk kalibrasi 8 detik, lalu relawan mengikuti instruksi jari yang
   tampil di layar. Total 100 percobaan (5 jari x 20 pengulangan) dengan urutan acak.
   Tiap percobaan berjalan seperti ini:
   - Relawan membuka tangan penuh dan menunggu minimal 0,8 detik (fase istirahat).
   - Jari target muncul. Pemicu pertama yang terdeteksi dalam 5 detik dicatat sebagai
     respons; jika tidak ada, dicatat -1 (tidak terdeteksi).
   - Pemicu tambahan dalam 0,6 detik setelah pemicu pertama ikut dihitung di kolom
     `jumlah_pemicu`.

   `Q` menghentikan pengujian, dan data yang sudah terekam tetap tersimpan di
   `hasil_uji/uji_<kode relawan>.csv` dengan kolom `relawan`, `trial`, `jari_target`,
   `jari_terdeteksi`, `benar`, `jumlah_pemicu`, `waktu_respon_s`, `fps`.
2. `python evaluasi.py` : membaca semua `hasil_uji/uji_*.csv`, lalu menampilkan
   confusion matrix, precision, recall, akurasi, nada keliru per sesi, waktu respon,
   dan FPS. Hasilnya juga disimpan ke `hasil_uji/ringkasan_evaluasi.csv` dan
   `hasil_uji/confusion_matrix.png` (gambar dibuat bila matplotlib terpasang).

Kode jari di CSV: 0 jempol, 1 telunjuk, 2 tengah, 3 manis, 4 kelingking, -1 tidak terdeteksi.

Definisi: percobaan dinyatakan benar bila pemicu pertama sama dengan jari target,
salah jari bila pemicu pertama jari lain, dan terlewat bila tidak ada pemicu dalam
5 detik. Nada keliru per percobaan adalah `jumlah_pemicu` dikurangi pemicu yang
benar, dijumlahkan per relawan lalu dirata-rata antar relawan. Rata-rata waktu respon
hanya dihitung dari percobaan yang terdeteksi.

## Hasil pengujian awal

Hasil dari satu relawan sehat (kode OR1, usia 20, tangan kanan), 100 percobaan:

| Metrik | Hasil |
|---|---|
| Akurasi keseluruhan | 83% (83 dari 100 percobaan) |
| Salah jari / terlewat | 13 / 4 |
| Precision per jari | 0,63 (jempol), 0,89 (telunjuk), 1,00 (tengah, manis, kelingking) |
| Recall per jari | 0,75 sampai 0,95 |
| Nada keliru per sesi | 37 |
| Rata-rata waktu respon | 1,68 detik |
| Rata-rata FPS | 32,9 |

Dari 13 percobaan salah jari, 11 terbaca sebagai jempol. Angka ini hasil satu sesi
dan akan berubah bila pengujian diulang atau relawan ditambah.

## Keterbatasan

- Pengujian baru melibatkan satu relawan sehat sehingga belum dapat digeneralisasi.
  Hasil ini mengukur kinerja teknis, bukan manfaat rehabilitasi klinis.
- Jempol cenderung terbaca saat jari lain digerakkan (precision 0,63). Penyebabnya
  masih dugaan dan belum diverifikasi.
- Uji kondisi diam (tangan rileks tanpa gerakan sengaja) belum dilakukan, sehingga
  pemicu tanpa gerakan belum terukur.
- Estimasi gerakan dari kamera 2D bersifat relatif, bukan sudut sendi sebenarnya.
  Nilai rentang gerak di ringkasan sesi berasal dari kalibrasi, bukan dipantau
  sepanjang sesi.
- Sistem saat ini melatih gerakan menekuk jari dan hanya mendeteksi satu tangan.

## Struktur file dan penanggung jawab

| File | Isi | Penanggung jawab |
|---|---|---|
| `hand_features.py` | Webcam, MediaPipe Hands, fitur tekukan jari | Mazaya |
| `calibration.py` | Kalibrasi buka-tutup tangan, ambang per jari | Eka |
| `temporal.py` | Smoothing (moving average) dan hysteresis | Eka |
| `audio.py` | Pembuatan dan pemutaran nada | Romanna |
| `hud.py` | Teks dan tampilan informasi di layar | Romanna |
| `main.py` | Aplikasi utama, tampilan, mode melodi, ringkasan sesi | Romanna |
| `mode_uji.py` | Pengumpulan data uji dari relawan | Lenggo |
| `evaluasi.py` | Confusion matrix, precision, recall, nada keliru, FPS | Lenggo |

## Dataset

Tidak memakai dataset publik. Data uji berupa log numerik sesi relawan sehat
(dengan persetujuan, identitas diganti kode). Tidak ada video yang disimpan.

## Anggota kelompok

- Romanna Tiolin Pangaribuan - 5026241024
- Putri Lenggo Genni - 5026241040
- Mazaya Zharfani Efindri - 5026241138
- Eka Destriana Putri -5026241164

## Tautan tambahan

- Video demo : https://drive.google.com/file/d/1dIPoKGuPg26vstayXBdThNbyZVB3hFUe/view?usp=sharing
