# 🍄 UAS Kecerdasan Buatan: Klasifikasi Jamur Beracun

## 📌 Judul Proyek  
**Perbandingan Algoritma Machine Learning: Decision Tree, Random Forest, dan Naive Bayes untuk Klasifikasi Jamur Beracun**

## 👥 Nama Kelompok  
1. **Gezant Ashabil Haqdu D** (2206049)  
2. **Moch Yusuf Ferandy** (2206038)

---

## 📁 Struktur Repositori

```
UAS_KecerdasanBuatan/
├── README.md                  # Penjelasan proyek (file ini)
├── uas_ai.md                  # Laporan UAS lengkap (versi markdown)
├── notebook_model.ipynb       # Implementasi kode & evaluasi model
└── data/
    ├── mushrooms.csv          # Dataset jamur dari UCI
    └── Jurnal/                # Referensi ilmiah (jika ada)
```
## 🎯 Business Understanding

### 📌 Permasalahan
Jamur merupakan salah satu sumber pangan yang umum dikonsumsi. Namun, sebagian jenis jamur bersifat beracun dan berbahaya jika dikonsumsi secara tidak sengaja. Identifikasi manual memerlukan keahlian mikologi yang tidak dimiliki oleh masyarakat awam atau pelaku industri.

### 🎯 Tujuan Proyek
1. Mengembangkan model klasifikasi jamur menggunakan tiga algoritma: Decision Tree, Random Forest, dan Naive Bayes.
2. Membandingkan performa ketiga model dalam mengklasifikasikan jamur beracun dan tidak beracun.
3. Mengidentifikasi fitur yang paling berpengaruh dalam proses klasifikasi.
4. Menyediakan solusi klasifikasi otomatis berbasis machine learning.

### 👤 Pengguna Sistem
- **Petani & komunitas pengumpul jamur**: untuk menghindari konsumsi jamur beracun.
- **Industri pangan**: sebagai bagian dari quality control bahan baku.
- **Peneliti biologi atau mikologi**: untuk analisis morfologi jamur secara cepat.

### 💡 Manfaat Implementasi AI
- Prediksi klasifikasi jamur secara cepat dan akurat.
- Menghindari risiko konsumsi jamur beracun.
- Memberikan insight terhadap fitur yang menentukan kategori jamur.
- Mendukung otomatisasi sistem klasifikasi tanpa perlu keahlian khusus.

## 📊 Data Understanding

### 📁 Sumber Data
- Dataset yang digunakan adalah **Mushroom Dataset** dari [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Mushroom).
- Dataset ini terdiri dari data nyata tentang fitur morfologi jamur dan label klasifikasi (edible atau poisonous).

### 🧬 Deskripsi Fitur
Dataset berisi **8124 baris** dan **23 kolom**, seluruhnya bertipe kategorikal.

| No | Kolom                    | Tipe Data | Keterangan Singkat                    |
|----|--------------------------|-----------|---------------------------------------|
| 1  | class                    | object    | Target: edible (`e`) atau poisonous (`p`) |
| 2  | cap-shape                | object    | Bentuk tudung jamur                  |
| 3  | cap-surface              | object    | Permukaan tudung                     |
| 4  | cap-color                | object    | Warna tudung                         |
| 5  | bruises                  | object    | Adakah memar pada jamur              |
| 6  | odor                     | object    | Bau jamur                            |
| 7  | gill-attachment          | object    | Jenis sambungan insang               |
| 8  | gill-spacing             | object    | Jarak antar insang                   |
| 9  | gill-size                | object    | Ukuran insang                        |
| 10 | gill-color               | object    | Warna insang                         |
| 11 | stalk-shape              | object    | Bentuk batang jamur                  |
| 12 | stalk-root               | object    | Jenis akar batang jamur              |
| 13 | stalk-surface-above-ring| object    | Permukaan batang atas cincin         |
| 14 | stalk-surface-below-ring| object    | Permukaan batang bawah cincin        |
| 15 | stalk-color-above-ring  | object    | Warna batang atas cincin             |
| 16 | stalk-color-below-ring  | object    | Warna batang bawah cincin            |
| 17 | veil-type                | object    | Jenis selaput                        |
| 18 | veil-color               | object    | Warna selaput                        |
| 19 | ring-number              | object    | Jumlah cincin pada batang            |
| 20 | ring-type                | object    | Jenis cincin                         |
| 21 | spore-print-color        | object    | Warna cetakan spora                  |
| 22 | population               | object    | Kelimpahan jamur                     |
| 23 | habitat                  | object    | Habitat tumbuh jamur                 |

### 📐 Ukuran dan Format
- **Format**: CSV
- **Jumlah Data**: 8124 entri
- **Tipe Data**: Kategorikal
- **Target Klasifikasi**: `class` (edible vs poisonous)

---

## ✅ Kesimpulan & Rekomendasi

- Semua model berhasil klasifikasikan jamur dengan akurat
- **Random Forest** direkomendasikan karena akurasi & stabilitas
- Rekomendasi:
  - Coba algoritma ensemble lain seperti XGBoost
  - Gunakan hyperparameter tuning
  - Uji pada dataset jamur lokal / lapangan

---

## 📚 Referensi

Tercantum lengkap di [uas_ai.md](./uas_ai.md), dengan minimal 5 jurnal ilmiah bergaya APA (contoh: Arslan et al., 2024; Ortega, 2020; Sulistianingsih et al., 2025).

---

## ▶️ Cara Menjalankan Kode

1. Clone repositori:
   ```bash
   git clone https://github.com/GezantAshabilHD/UAS_KecerdasanBuatan.git
   cd UAS_KecerdasanBuatan

## ▶️ Cara Menjalankan Notebook

1. Buka file `notebook_model.ipynb` di Google Colab atau Jupyter Notebook.
2. Jalankan semua sel secara berurutan dari atas ke bawah.

---

# 📊 Laporan UAS Kecerdasan Buatan
## Perbandingan Algoritma Decision Tree, Random Forest, dan Naive Bayes pada Klasifikasi Jamur Beracun

### 👥 Nama Kelompok
1. Gezant Ashabil Haqdu D (2206049)  
2. Moch Yusuf Ferandy (2206038)


📝 **Catatan**:  
Proyek ini dibuat untuk memenuhi tugas akhir mata kuliah **Kecerdasan Buatan**. Dikerjakan menggunakan pendekatan *real-world project* dengan dukungan referensi ilmiah yang valid.

