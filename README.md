# Instrumen Musik Udara untuk Rehabilitasi Motorik Jari Pascastroke

Implementasi Computer Vision untuk rehabilitasi motorik jari pasien pascastroke
melalui instrumen musik udara menggunakan OpenCV dan MediaPipe.

Gerakan jari dideteksi lewat webcam laptop dan diubah menjadi nada (C4 sampai G4,
satu nada per jari). Sistem juga mencatat ringkasan sesi latihan (repetisi,
rentang gerak relatif, usaha maksimum, ketepatan melodi).
Sistem ini alat bantu latihan, bukan alat medis. Tidak ada video yang disimpan.

## Fitur

- Deteksi tangan dengan MediaPipe Hands (21 landmark) dari webcam. Jika dua tangan
  terlihat, aplikasi memilih tangan yang pergelangannya paling dekat dengan tangan
  di frame sebelumnya dan menampilkan peringatan.
- Dua metode fitur tekukan jari, dipilih dengan `--fitur`:
  - `jarak` (default): jarak ujung jari ke pergelangan dibagi ukuran telapak.
  - `sudut`: sudut sendi jari dalam derajat (jempol memakai sendi IP, jari lain
    memakai sendi PIP), dihitung dari 3 landmark sehingga jempol tidak bergantung
    pada posisi jari lain.
- Kalibrasi dua fase per pengguna: posisi istirahat (4 detik), lalu rentang gerak
  (8 detik). Usaha tiap jari dihitung relatif terhadap posisi istirahat pengguna,
  dengan zona mati dari noise saat istirahat. Jari yang geraknya terlalu kecil
  dinonaktifkan, tidak menggagalkan seluruh kalibrasi.
- Dua arah latihan: TUTUP (nada saat jari menekuk dari posisi istirahat) dan BUKA
  (nada saat jari meluruskan dari posisi istirahat). Arah dipilih otomatis setelah
  kalibrasi dan bisa diganti, sehingga tangan kaku yang cenderung menggenggam tetap
  bisa berlatih.
- Smoothing (rata-rata bergerak 5 frame) dan hysteresis (picu pada usaha 0,6 ke atas,
  lepas pada 0,35 ke bawah, sensitivitas bisa diubah) agar satu gerakan dihitung
  satu repetisi. Tangan yang sudah menggenggam saat mulai tidak langsung membunyikan nada.
- Mode bebas, mode melodi dengan 3 level, dan tugas seluruh tangan (satu akord per ulangan).
- Peringatan di layar: dua tangan terlihat, tangan terlalu kecil (terlalu jauh dari
  kamera), tangan tidak terdeteksi, dan pengingat istirahat setiap 5 menit latihan aktif.
- Ringkasan sesi di layar dan ekspor ke CSV.

## Cara menjalankan

1. Gunakan Python 3.12.10. Versi MediaPipe dikunci di requirements.txt, jangan di-upgrade.
2. Buat dan aktifkan virtual environment:
   - Windows: `python -m venv venv` lalu `venv\Scripts\activate`
   - Mac/Linux: `python -m venv venv` lalu `source venv/bin/activate`
3. Install dependensi: `pip install -r requirements.txt`
4. Hubungkan webcam, lalu jalankan: `python main.py`

Opsi: `--kamera N` untuk memilih kamera (default 0, coba 1 jika kamera tidak terbuka)
dan `--fitur jarak|sudut` untuk memilih metode fitur, misalnya `python main.py --fitur sudut`.

Folder `venv/` jangan ikut di-commit atau di-zip (sudah ada di `.gitignore`). Buat ulang
di komputer masing-masing dengan langkah di atas.

## Cara memakai aplikasi (`main.py`)

1. Saat aplikasi terbuka, tekan `C`. Kalibrasi berjalan dua fase:

   - Fase 1 (4 detik): istirahatkan tangan dan jangan digerakkan.
   - Fase 2 (8 detik): gerakkan jari semampunya (buka-tutup).

   Jika kalibrasi gagal (tangan kurang terdeteksi, atau tidak ada jari yang bergerak
   cukup jelas dibanding noise), muncul pesan di layar dan kalibrasi bisa diulang dengan `C`.
   Jari yang dinonaktifkan ditampilkan di pesan setelah kalibrasi berhasil.
2. Setelah kalibrasi berhasil, arah latihan dipilih otomatis (arah dengan jari aktif
   terbanyak, seri dipilih TUTUP). Gerakkan jari untuk membunyikan nadanya (mode bebas).
3. Tekan `M` untuk mode melodi, `D` untuk mengganti level, `H` untuk tugas seluruh tangan.
4. Tekan `Q` untuk keluar. Ringkasan sesi tampil dan disimpan.

| Tombol                   | Fungsi                                                                     |
| ------------------------ | -------------------------------------------------------------------------- |
| `C`                    | Kalibrasi dua fase (istirahat 4 detik, gerak 8 detik)                      |
| `B`                    | Ganti arah latihan TUTUP/BUKA (setelah kalibrasi, jika arah lain tersedia) |
| `H`                    | Tugas seluruh tangan hidup/mati (setelah kalibrasi)                        |
| `M`                    | Mode melodi hidup/mati (setelah kalibrasi)                                 |
| `D`                    | Ganti level melodi (setelah kalibrasi)                                     |
| `T`                    | Melodi ketat/toleran                                                       |
| `A`                    | Tandai sesi DIBANTU/MANDIRI                                                |
| `P`                    | Jeda/lanjut (setelah kalibrasi)                                            |
| `[` / `]`            | Sensitivitas lebih mudah / lebih sulit memicu (rentang 0,30 sampai 0,80)   |
| `+` atau `=` / `-` | Volume naik / turun                                                        |
| `Q`                    | Keluar, simpan, dan tampilkan ringkasan sesi                               |

### Mode melodi

| Level           | Jari dalam melodi                 | Batas waktu per nada |
| --------------- | --------------------------------- | -------------------- |
| Mudah           | 3 jari (jempol, telunjuk, tengah) | Tanpa batas          |
| Sedang          | 5 jari                            | Tanpa batas          |
| Tantangan waktu | 5 jari                            | Adaptif              |

Batas waktu adaptif adalah 3 kali median waktu benar terakhir, dibatasi 8 sampai 30 detik
(awalnya 12 detik). Jari yang nonaktif dilewati dari melodi. Jika semua jari melodi
nonaktif, mode melodi dimatikan.

Jari target ditandai kotak kuning pada batang usaha. Jika jari yang digerakkan benar,
nada berbunyi, melodi maju, dan layar berkedip hijau. Nada yang melewati batas waktu
dihitung terlewat, layar berkedip merah, dan melodi lanjut ke nada berikutnya.

Perilaku jari yang salah bergantung pada mode (tombol `T`):

- Toleran (default): jari lain yang ikut bergerak diabaikan tanpa hukuman.
- Ketat: jari yang salah dihitung salah, terdengar nada rendah yang kasar, dan layar
  berkedip merah. Melodi tidak maju.

### Tugas seluruh tangan

Tekan `H`, lalu gerakkan semua jari aktif bersama. Satu ulangan dihitung saat rata-rata
usaha jari aktif melewati ambang, dan terdengar akord C-E-G. Jumlah ulangan tampil di
layar dan masuk ringkasan sesi. Mode ini dan mode melodi tidak bisa aktif bersamaan.

### Ringkasan sesi

Setelah `Q`, ringkasan sesi tampil di layar (tutup dengan menekan tombol apa saja)
dan disimpan ke `hasil_sesi/ringkasan_sesi.csv`, dengan syarat ada minimal satu
repetisi jari atau satu ulangan seluruh tangan pada sesi tersebut. Isinya:

- durasi sesi, repetisi per jari, dan ulangan seluruh tangan,
- rentang gerak relatif per jari (dari kalibrasi) dan usaha maksimum per jari selama sesi (0 sampai 1),
- indeks isolasi jari (1 = hanya jari target yang bergerak, makin kecil berarti jari lain ikut bergerak),
- benar, salah, terlewat, dan akurasi melodi per level,
- penanda sesi dibantu/mandiri, arah latihan, jari aktif, dan metode fitur,
- persen frame tangan terdeteksi, persen frame dua tangan, dan persen frame tangan terlalu kecil.

Jika format kolom file lama berbeda, file lama diganti nama menjadi
`ringkasan_sesi_lama_<waktu>.csv` sebelum file baru dibuat. Folder `hasil_sesi/`
tercantum di `.gitignore` sehingga tidak ikut repositori.

## Pengujian dan evaluasi

Urutan yang disarankan: cek deteksi postur dulu, lalu uji respons jari, lalu uji kondisi diam.
Kalau tangan menggenggam sering tidak terdeteksi, perbaikan lain tidak banyak berguna.

Semua skrip uji meminta kode relawan di terminal (misalnya `R1`, identitas diganti kode),
memakai kalibrasi dua fase yang sama dengan `main.py`, dan menyimpan data numerik saja ke `hasil_uji/`.
Opsi `--kamera`, `--fitur`, `--arah auto|tutup|buka`, dan `--skenario` (label kondisi uji,
misalnya `normal`, `kaku_sedang`, `kaku_kecil`, `genggam`) tersedia sesuai skrip.

### 1. Cek deteksi postur (`uji_deteksi_postur.py`)

`python uji_deteksi_postur.py [--detik 8] [--kamera 0]`

Tahan tiga postur masing-masing 8 detik (genggam, buka sebagian, buka lebar), tekan `SPASI`
untuk memulai tiap postur. Keluaran di terminal dan `hasil_uji/postur_<relawan>.csv`:
persen frame tangan terdeteksi per postur, rata-rata dan simpangan tiap jari untuk fitur
`jarak` dan `sudut`, serta pemisahan genggam vs terbuka (selisih rata-rata dibagi simpangan
gabungan). Makin besar pemisahan, makin jelas fitur membedakan kedua postur. Di bawah sekitar 2
berarti tumpang-tindih dengan noise.

### 2. Uji respons jari (`mode_uji.py`)

`python mode_uji.py [--skenario normal] [--arah auto] [--fitur jarak] [--ulang 20]`

Tekan `SPASI` untuk kalibrasi, lalu relawan mengikuti instruksi jari yang tampil di layar.
Jumlah percobaan adalah jumlah jari aktif dikali `--ulang` (default 20), jadi 100 percobaan
jika kelima jari aktif, dengan urutan acak. Tiap percobaan berjalan seperti ini:

- Relawan kembali ke posisi istirahat dan menunggu minimal 0,8 detik (fase istirahat).
- Jari target muncul. Pemicu pertama yang terdeteksi dalam `--timeout` (default 5 detik)
  dicatat sebagai respons; jika tidak ada, dicatat -1 (tidak terdeteksi).
- Pemicu tambahan dalam `--jendela` (default 0,6 detik) setelah pemicu pertama ikut
  dihitung di kolom `jumlah_pemicu`.

Nilai default `--timeout` dan `--jendela` sama dengan laporan agar hasil tetap sebanding.
Jika diubah, laporkan sebagai kondisi uji yang terpisah.

`Q` menghentikan pengujian, dan data yang sudah terekam tetap tersimpan di
`hasil_uji/uji_<relawan>_<skenario>_<arah>_<fitur>.csv` dengan kolom `relawan`, `trial`,
`jari_target`, `jari_terdeteksi`, `benar`, `jumlah_pemicu`, `waktu_respon_s`, `fps`,
`skenario`, `arah`, `fitur`, `frame_terdeteksi_persen`, `isolasi`, `timeout_s`.

### 3. Uji kondisi diam (`uji_diam.py`)

`python uji_diam.py [--durasi 60] [--skenario normal] [--arah auto] [--fitur jarak]`

Tekan `SPASI` untuk kalibrasi, lalu relawan merilekskan tangan dan tidak menggerakkannya
selama durasi uji. Setiap pemicu yang terjadi dihitung sebagai pemicu palsu. Target di
rencana laporan adalah paling banyak 1 pemicu per menit. Hasil disimpan di
`hasil_uji/diam_<relawan>_<skenario>_<arah>_<fitur>.csv` (total pemicu, pemicu per menit,
persen frame terdeteksi, dan pemicu per jari).

### 4. Evaluasi (`evaluasi.py`)

`python evaluasi.py`

Membaca semua `hasil_uji/uji_*.csv` dan mengelompokkannya per kondisi uji (skenario, arah,
metode fitur) agar hasil tidak tercampur. File lama tanpa kolom skenario dianggap
skenario `normal`, arah `tutup`, fitur `jarak`. Untuk tiap kondisi ditampilkan confusion matrix,
precision, recall, akurasi, nada keliru per sesi, waktu respon, FPS, persen frame terdeteksi,
dan indeks isolasi. Jari yang tidak diuji ditampilkan `n/a`. Hasil uji diam (`diam_*.csv`)
ikut diringkas. Keluaran disimpan ke `hasil_uji/ringkasan_evaluasi.csv` dan
`hasil_uji/confusion_matrix_<skenario>_<arah>_<fitur>.png` (gambar dibuat bila matplotlib
terpasang). Jika hanya ada satu kondisi, `hasil_uji/confusion_matrix.png` juga ikut dibuat
atau ditimpa.

Kode jari di CSV: 0 jempol, 1 telunjuk, 2 tengah, 3 manis, 4 kelingking, -1 tidak terdeteksi.

Definisi: percobaan dinyatakan benar bila pemicu pertama sama dengan jari target, salah jari
bila pemicu pertama jari lain, dan terlewat bila tidak ada pemicu dalam batas waktu.
Nada keliru per percobaan adalah `jumlah_pemicu` dikurangi pemicu yang benar, dijumlahkan
per relawan lalu dirata-rata antar relawan. Rata-rata waktu respon hanya dihitung dari
percobaan yang terdeteksi.

## Hasil pengujian

Data diambil dengan versi kode terbaru (kalibrasi dua fase, fitur `jarak`, arah TUTUP,
skenario `normal`) dari tiga relawan sehat (kode OR2, OR3, OR4), masing-masing 100 percobaan
(5 jari x 20 percobaan, urutan acak), total 300 percobaan. Data mentah ada di
`hasil_uji/uji_OR*_normal_tutup_jarak.csv`, ringkasannya di `hasil_uji/ringkasan_evaluasi.csv`.

### Uji respons jari

| Metrik                         | Hasil                                                                          |
| ------------------------------ | ------------------------------------------------------------------------------ |
| Akurasi keseluruhan            | 94,7% (284 dari 300 percobaan)                                                 |
| Akurasi per relawan            | OR2 94%, OR3 96%, OR4 94%                                                      |
| Salah jari / terlewat          | 12 / 4                                                                         |
| Precision per jari             | 0,92 (jempol), 0,93 (telunjuk), 0,97 (tengah), 0,98 (manis), 1,00 (kelingking) |
| Recall per jari                | 0,88 (manis), 0,95 (telunjuk, tengah, kelingking), 1,00 (jempol)               |
| Nada keliru per sesi           | 8,67 (OR2 17, OR3 5, OR4 4)                                                    |
| Rata-rata waktu respon         | 1,48 detik (termasuk waktu reaksi relawan, bukan latensi sistem murni)         |
| Rata-rata FPS                  | 19,9                                                                           |
| Persen frame tangan terdeteksi | 100%                                                                           |
| Indeks isolasi                 | 0,957                                                                          |

Dari 12 percobaan salah jari, 5 terbaca sebagai jempol, 4 telunjuk, 2 tengah, dan 1 manis.

### Uji kondisi diam

Satu relawan (OR4), 60 detik, tangan rileks: 0 pemicu palsu (0 per menit), di bawah target
paling banyak 1 pemicu per menit. Frame tangan terdeteksi 100%.

### Cek deteksi postur

Satu relawan (OR2): tangan terdeteksi 100% pada ketiga postur (genggam, buka sebagian,
buka lebar). Pemisahan genggam vs terbuka untuk fitur `jarak` berkisar 2,51 sampai 3,12
(di atas batas sekitar 2), sedangkan untuk fitur `sudut` 2,25 sampai 2,75.

Semua angka berasal dari sampel kecil dan akan berubah bila pengujian diulang atau
relawan ditambah.

## Keterbatasan

- Pengujian respons jari melibatkan tiga relawan sehat, sedangkan uji kondisi diam dan cek
  postur baru satu relawan (satu sesi). Belum ada pengujian pada pasien pascastroke, sehingga
  hasil belum dapat digeneralisasi. Hasil ini mengukur kinerja teknis, bukan manfaat
  rehabilitasi klinis.
- Waktu respon mencakup waktu reaksi relawan terhadap instruksi, sehingga bukan ukuran
  kecepatan sistem saja. FPS bergantung pada spesifikasi laptop yang dipakai.
- Skenario tangan kaku (misalnya `kaku_sedang`, `kaku_kecil`, `genggam`) tersedia di skrip uji
  sebagai tiruan oleh relawan sehat, tetapi belum diuji. Data yang ada hanya skenario `normal`
  arah TUTUP dengan fitur `jarak`.
- Jempol memiliki precision terendah (0,92) dan paling sering terbaca saat jari lain
  digerakkan. Penyebabnya masih dugaan dan belum diverifikasi. Metode fitur `sudut` dibuat agar
  jempol tidak bergantung pada posisi jari lain, tetapi pada uji respons jari efeknya belum diuji.
- Estimasi gerakan dari kamera 2D bersifat relatif, bukan sudut sendi sebenarnya. Rentang
  gerak di ringkasan sesi berasal dari kalibrasi, sedangkan usaha maksimum dipantau selama
  sesi tetapi tetap relatif terhadap kalibrasi.
- Sistem saat ini melatih gerakan menekuk atau meluruskan jari dan hanya memakai satu tangan.

## Struktur file dan penanggung jawab

| File                      | Isi                                                                    | Penanggung jawab |
| ------------------------- | ---------------------------------------------------------------------- | ---------------- |
| `hand_features.py`      | Webcam, MediaPipe Hands, fitur tekukan jari (jarak dan sudut)          | Mazaya           |
| `calibration.py`        | Kalibrasi dua fase, usaha relatif, zona mati, validitas jari per arah  | Eka              |
| `temporal.py`           | Smoothing (moving average) dan hysteresis                              | Eka              |
| `audio.py`              | Pembuatan dan pemutaran nada dan akord                                 | Romanna          |
| `hud.py`                | Teks dan tampilan informasi di layar                                   | Romanna          |
| `main.py`               | Aplikasi utama, tampilan, mode melodi, seluruh tangan, ringkasan sesi  | Romanna          |
| `mode_uji.py`           | Pengumpulan data uji respons jari dari relawan                         | Lenggo           |
| `evaluasi.py`           | Confusion matrix, precision, recall, nada keliru, FPS, per kondisi uji | Lenggo           |
| `uji_deteksi_postur.py` | Cek deteksi postur dan pemisahan fitur                                 | Mazaya           |
| `uji_diam.py`           | Uji pemicu palsu pada kondisi diam                                     | Mazaya           |

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
