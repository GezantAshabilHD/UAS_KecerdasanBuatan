# Laporan UAS Kecerdasan Buatan
# Judul Proyek
Perbandingan Algoritma Machine Learning Menggunakan Decision Tree, Random Forest, dan Naive Bayes pada Klasifikasi Jamur Beracun

# Nama Kelompok
1. Gezant Ashabil Haqdu D  (2206049)
2. Moch Yusuf Ferandy      (2206038)

# Domain Proyek
Klasifikasi jamur beracun dan tidak beracun merupakan hal penting dalam bidang pertanian dan keamanan pangan. Salah identifikasi dapat berakibat fatal bagi kesehatan manusia. Dalam proyek ini, diterapkan algoritma Decision Tree, Random Forest, dan Naive Bayes untuk membandingkan performa dalam mengklasifikasikan jenis jamur berdasarkan atribut biologisnya (Sulistianingsih et al., 2025). Tujuannya adalah mencari metode paling efektif dan akurat berdasarkan dataset jamur dari UCI (Hamonangan et al., 2021).

# Business Understanding
## Problem Statements
Jamur merupakan salah satu sumber pangan yang umum dikonsumsi, namun beberapa jenis di antaranya bersifat beracun dan dapat menyebabkan keracunan serius hingga kematian apabila dikonsumsi secara tidak sengaja. Di sisi lain, klasifikasi jenis jamur memerlukan keahlian mikologi yang tidak dimiliki oleh kebanyakan masyarakat awam atau pelaku industri pangan. Oleh karena itu, penggunaan metode machine learning menjadi alternatif solusi yang cepat dan akurat untuk mengklasifikasikan jamur berdasarkan karakteristik fisiknya. Namun, tidak semua algoritma memiliki performa yang sama. Diperlukan perbandingan algoritma seperti Decision Tree, Random Forest, dan Naive Bayes untuk mengetahui metode terbaik dalam klasifikasi jamur beracun dan tidak beracun (Sulistianingsih et al., 2025).

## Goals
Beberapa tujuan utama dari proyek ini adalah:
1. Mengembangkan model klasifikasi jamur menggunakan tiga algoritma machine learning: Decision Tree, Random Forest, dan Naive Bayes.
2. Membandingkan performa ketiga model berdasarkan metrik evaluasi seperti akurasi, precision, recall, dan F1-score.
3. Mengidentifikasi fitur-fitur yang paling berpengaruh dalam klasifikasi jamur beracun dan tidak beracun.
4. Menentukan algoritma yang paling efektif dan efisien untuk digunakan dalam sistem klasifikasi otomatis berbasis data.
5. Mengimplementasikan pipeline klasifikasi dari preprocessing hingga evaluasi menggunakan Python dan pustaka scikit-learn.

## System User
Pengguna sistem dalam penelitian ini adalah pihak-pihak yang memiliki kepentingan terhadap keamanan konsumsi jamur serta efisiensi dalam proses klasifikasinya. Beberapa kelompok pengguna potensial meliputi:

1. **Petani dan komunitas pengumpul jamur**  
Pemilihan algoritma yang tepat sangat penting untuk memastikan prediksi risiko diabetes memiliki akurasi tinggi dengan performa yang optimal.

2. **Industri pangan dan pengolahan hasil pertanian**    
Sistem ini dapat digunakan sebagai bagian dari quality control untuk memastikan bahwa hanya jamur yang aman dan dapat dikonsumsi yang diproses lebih lanjut.

3. **Lembaga riset pertanian atau biologi**  
Peneliti dapat menggunakan sistem ini untuk menguji hipotesis, mengklasifikasikan spesies jamur, atau sebagai referensi tambahan dalam identifikasi morfologi jamur

## Benefits of AI Implementation
Beberapa manfaat dari implementasi AI menggunakan algoritma Decision Tree, Random Forest, dan Naive Bayes pada klasifikasi jamur beracun berdasarkan hasil dari penelitian dan eksperimen di Google Colab antara lain:  
1. Memberikan prediksi klasifikasi jamur secara cepat dan akurat berdasarkan fitur morfologi.
2. Memudahkan proses identifikasi jamur tanpa perlu keahlian mikologi, cukup dengan input data karakteristik jamur.
3. Menunjukkan keunggulan Random Forest dari sisi akurasi dan stabilitas prediksi, sesuai hasil evaluasi model di Google Colab.
4. Memberikan insight terhadap fitur-fitur yang paling berpengaruh dalam membedakan jamur beracun dan tidak beracun.
5. Mendukung pengambilan keputusan dalam pengawasan kualitas hasil panen jamur atau konsumsi jamur liar

# Data Understanding
## Sumber Data
Dataset yang digunakan adalah Mushroom Dataset dari UCI Machine Learning Repository. Dataset ini telah banyak dimanfaatkan dalam studi klasifikasi jamur karena terdiri dari data nyata tentang fitur morfologi jamur dan label klasifikasi aman atau beracun. Dataset ini juga digunakan dalam penelitian oleh Arslan et al. (2024), yang melakukan perbandingan beberapa metode machine learning untuk optimasi klasifikasi jamur.
Dataset yang digunakan adalah Mushroom Dataset dari UCI Machine Learning Repository yang telah digunakan pada banyak studi sebelumnya (Sulistianingsih et al., 2025).

## Deskripsi Fitur
Dataset terdiri dari 8124 entri dan 22 atribut kategorikal yang menggambarkan ciri-ciri fisik jamur.  

| NO |      column               |    Non-Null Count  |  Tipe Data   |    
|----|---------------------------|--------------------|--------------|
| 0  | class                     |         8142       |    object    |
| 1  | cap-shape                 |         8142       |    object    |
| 2  | cap surface               |         8142       |    object    |                         
| 3  | cap-color                 |         8142       |    object    |
| 4  | bruises                   |         8142       |    object    |
| 5  | odor                      |         8142       |    object    | 
| 6  | gill-attachment           |         8142       |    object    |
| 7  | giLL-spacing              |         8142       |    object    | 
| 8  | gill-size                 |         8142       |    object    |
| 9  | gill-color                |         8142       |    object    |
| 10 | stalk-shape               |         8142       |    object    |
| 11 | stalk-root                |         8142       |    object    |
| 12 | stalk-surface-above-ring  |         8142       |    object    |
| 13 | stalk-surface-below-ring  |         8142       |    object    | 
| 14 | stalk-color-above-ring    |         8142       |    object    |
| 15 | stalk-color-below-ring    |         8142       |    object    | 
| 16 | veil-type                 |         8142       |    object    |
| 17 | veil-color                |         8142       |    object    |
| 18 | ring-number               |         8142       |    object    |
| 19 | ring-type                 |         8142       |    object    |
| 20 | spore-pirnt-color         |         8142       |    object    |  
| 21 | population                |         8142       |    object    |  
| 22 | habitat                   |         8142       |    object    |

## Ukuran dan Format Data
1. Jumlah entri: 8124
2. Jumlah fitur: 22 fitur kategorikal + 1 label (poisonous/edible)

# Exploratory Data Analysis (EDA)  
## Visualisasi Distribusi Label
Visualisasi data menunjukkan bahwa kelas edible dan poisonous memiliki distribusi yang cukup seimbang dalam dataset. Hal ini mempermudah proses pelatihan karena model tidak mengalami bias akibat ketidakseimbangan data. Visualisasi dilakukan menggunakan plot batang dari nilai target setelah label encoding. Dalam eksperimen Google Colab, grafik distribusi label juga memperlihatkan bahwa masing-masing kelas berada pada jumlah yang hampir sama (Pokhrel, 2024). Korelasi antar fitur kategorikal dianalisis melalui metode statistik atau visualisasi seperti heatmap berdasarkan label encoding.

## Korelasi Antar Fitur
Dalam dataset ini, semua fitur bersifat kategorikal. Korelasi antar fitur dievaluasi melalui pengaruhnya terhadap label menggunakan metode feature importance dari model decision tree. Fitur seperti odor, gill-color, dan spore-print-color menunjukkan korelasi tinggi terhadap label kelas. Hasil ini sejalan dengan temuan dari penelitian sebelumnya yang menyatakan bahwa aroma jamur adalah penentu kuat dalam klasifikasi (Arslan et al., 2024).

## Deteksi Kelas Tidak Seimbang 
Keseimbangan ini menguntungkan karena tidak memerlukan penyesuaian data seperti oversampling atau undersampling. Oleh karena itu, semua algoritma dapat digunakan langsung tanpa modifikasi data tambahan
Deteksi Kelas Tidak Seimbang
Tidak terdapat ketidakseimbangan signifikan dalam distribusi kelas, sehingga tidak diperlukan teknik balancing data

## Insight Awal
Beberapa fitur memiliki kontribusi penting dalam klasifikasi jamur, khususnya odor, yang hampir secara langsung dapat menentukan apakah jamur termasuk beracun atau tidak. Selain itu, kombinasi fitur morfologis lainnya seperti ring-type dan gill-size juga memiliki pengaruh signifikan terhadap hasil klasifikasi. Hal ini memperkuat temuan sebelumnya bahwa fitur-fitur visual dan biologis jamur memiliki korelasi kuat terhadap label (Arslan et al., 2024).

# Data Preparation
## Langkah-langkah  

1. **Menghapus Fitur Tidak Relevan**  
Tidak semua fitur di dataset memiliki kontribusi signifikan. Misalnya, veil-type hanya memiliki satu nilai unik dan dihapus dari dataset.  

2. **Penanganan Missing Value**  
Fitur stalk-root memiliki nilai yang kosong ditandai dengan “?”. Dalam program Google Colab, nilai ini ditangani dengan menggantinya menggunakan metode mode imputation (mengisi dengan nilai terbanyak).  

3. **Label Encoding**  
Seluruh fitur dalam dataset bersifat kategorikal. Oleh karena itu, digunakan teknik Label Encoding untuk mengubah nilai string menjadi numerik agar dapat diproses oleh algoritma Decision Tree, Random Forest, dan Naive Bayes.  

4. **Split Dataset**  
Dataset dibagi menjadi data latih dan data uji dengan rasio 80:20 menggunakan fungsi train_test_split dari library sklearn.model_selection. Hal ini dilakukan untuk mengukur performa model secara objektif terhadap data yang belum pernah dilihat.  

5. **Pemisahan Fitur dan Target**  
Data dipisahkan antara variabel input (X) dan label target (y) sebelum dilakukan proses pelatihan model.

# Modelling
## Algoritma yang Digunakan  

1. **Decision Tree**  
Menghasilkan pohon keputusan berdasarkan pemilihan atribut terbaik secara rekursif.  

2. **Random Forest**  
Menggabungkan beberapa pohon keputusan untuk meningkatkan akurasi.  

3. **Naive Bayes**  
Menggunakan pendekatan probabilistik dengan asumsi independensi antar fitur.

## Alasan Pemilihan  
Ketiga algoritma tersebut banyak digunakan untuk klasifikasi data kategorikal dan memberikan hasil yang kompetitif pada kasus klasifikasi jamur (Sulistianingsih et al., 2025).  

## Implementasi  
Model diimplementasikan menggunakan scikit-learn dan dievaluasi dengan data uji  

# Evaluation
## Confusion Matrix
Confusion matrix digunakan untuk membandingkan hasil prediksi terhadap nilai aktual pada masing-masing model (C. Ortega, 2020).  

## Matrix Evaluasi  
Berikut adalah hasil evaluasi ketiga model berdasarkan akurasi, precision, recall, dan F1-score yang diperoleh dari hasil program di Google Colab:  

**Perbandingan Performa Model**  

|   Model       | Accuracy | Precision | Recall | F1- Score |
|---------------|----------|-----------|--------|-----------|
| Decision Tree | 1.0000   | 1.0000    | 1.0000 | 1.0000    |
| Random Forest | 1.0000   | 1.0000    | 1.0000 | 1.0000    |
| Naive Bayes   | 0.9218   | 0,9099    | 0.9297 | 0.9190    |  

![Perbandingan Model F1-Score](https://github.com/GezantAshabilHD/UAS_KecerdasanBuatan/blob/main/Perbandingan%20Model%20F1-Score.png?raw=true)

## Interpretasi
Random Forest biasanya memberikan hasil terbaik karena ensemble dari banyak pohon yang mengurangi overfitting 
Naive Bayes sangat cepat, tetapi memiliki keterbatasan karena asumsi independensi fitur 

# Kesimpulan dan Rekomendasi
## Ringkasan  
Semua model berhasil memprediksi kelas jamur dengan tingkat akurasi tinggi. Namun, Random Forest memberikan hasil terbaik secara umum 

## Tujuan Proyek
Tujuan untuk membandingkan tiga algoritma klasifikasi berhasil dicapai.

## Kelebihan
1. Decision Tree mudah diinterpretasi
2. Random Forest akurat dan stabil 
3. Naive Bayes efisien secara komputasi

## Keterbatasan
1. Asumsi independensi pada Naive Bayes
2. Interpretasi sulit pada Random Forest

## Rekomendasi
1. Uji kombinasi fitur
2. Eksperimen dengan tuning parameter dan ensemble lainnya

# Referensi (APA Style)
- Arslan, M., Azam, M., Ali, M., Hashmi, M. U., & Kousar, A. (2024). A Comparative Study of Machine Learning Methods for Optimizing Mushroom Classification. 08(01).
- C. Ortega, J. H. J. (2020). Analysis of Performance of Classification Algorithms in Mushroom Poisonous Detection using Confusion Matrix Analysis. International Journal of Advanced Trends in Computer Science and Engineering, 9(1.3), 451–456. https://doi.org/10.30534/ijatcse/2020/7191.32020
- Hamonangan, R., Saputro, M. B., Bagus, C., Dinata, S., & Atmaja, K. (2021). Accuracy of classification poisonous or edible of mushroom using naïve bayes and k-nearest neighbors. Journal of Soft Computing Exploration, 2(1). https://doi.org/10.52465/joscex.v2i1.26
- Metlek, S., & Çetiner, H. (2023). Classification of Poisonous and Edible Mushrooms with Optimized Classification Algorithms. International Conference on Applied Engineering and Natural Sciences, 1(1), 408–415. https://doi.org/10.59287/icaens.1030
- Sulistianingsih, N., Martono, G. H., Program, M., Bumigora, U., Learning, E., Forest, R., & Classifier, V. (2025). Analysis of the Effectiveness of Traditional and Ensemble Machine Learning Models for Mushroom Classification. 10, 48–60.
- Pokhrel, S. (2024). No TitleΕΛΕΝΗ. Αγαη, 15(1), 37–48. https://doi.org/10.5281/zenodo.754746

