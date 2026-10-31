# Simulator Mesin 3D – Mobil Sport

A sports car made of glowing points that you rotate and take apart (exploded view)
with hand gestures through your webcam. Built with Python, OpenCV and MediaPipe.

## Gerakan tangan

| Gerakan | Hasil |
|---|---|
| **Dua tangan dijauhkan** | Mobil terbongkar: bodi, kaca, mesin V8, transmisi, roda, rem, suspensi, jok, setir, knalpot, radiator, rangka |
| **Dua tangan didekatkan** | Mobil terakit kembali |
| **Dua tangan dimiringkan** (seperti memutar setir) | Mobil berputar |
| **Satu tangan terbuka digeser** | Putar mobil kiri-kanan / atas-bawah |
| **Cubit** (jempol + telunjuk) sebuah bagian, lalu tarik | Bagian itu ikut tertarik keluar mengikuti jari |
| Lepas cubitan | Bagian tetap di tempat barunya (bisa diulang untuk bagian lain) |
| **Satu tangan mengepal** | Mobil terakit kembali, semua bagian yang ditarik kembali ke tempatnya |
| Tanpa tangan | Mobil berputar pelan |

Saat terbongkar, nama setiap bagian muncul dengan garis penunjuk.
Bagian yang sedang dicubit menyala dan namanya tampil berwarna emas.
Paling mudah mencubit saat mobil sudah dibongkar, karena bagian-bagiannya terpisah.

## Isi folder

```
mesin3d_project/
├── simulator_mesin.py     program utama (jalankan ini)
├── model_mobil.py         model 3D mobil dari titik-titik (bisa diubah)
├── requirements.txt
└── hand_landmarker.task   model tangan MediaPipe (salin dari proyek mitologi)
```

## Menjalankan (Mac, Terminal VS Code)

```bash
cd folder/mesin3d_project
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python simulator_mesin.py
```

## Tombol

| Tombol | Fungsi |
|---|---|
| L | Tampilkan/sembunyikan nama bagian |
| H | Tampilkan/sembunyikan titik tangan |
| I | Info: jumlah tangan, fps, dan bentuk tangan yang terbaca (pinch/fist/open + angka jepit) |
| R | Reset posisi |
| F | Layar penuh / jendela |
| Q atau Esc | Keluar |

## Mengubah model

Setiap bagian mobil dibuat di `build_car()` dalam `model_mobil.py`:
nama, titik-titik, warna, dan arah bongkarnya (`dir`). Ubah angka `dir`
supaya bagian terlempar lebih jauh atau ke arah lain.
