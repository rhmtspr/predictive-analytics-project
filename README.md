# Laporan Proyek Machine Learning - Rachmat Risky Saputra

## Domain Proyek

Domain yang saya pilih untuk proyek predictive analysis menggunakan machine learning ini adalah di bidang pertanian. Dengan judul: Klasifikasi Kualitas Buah Anggur.

### Latar Belakang

Buah anggur termasuk salah satu buah yang cukup digemari di berbagai belahan dunia. Selain rasanya yang enak, anggur juga praktis karena bisa langsung dimakan tanpa harus dikupas terlebih dahulu. Tanaman anggur berasal dari wilayah Amerika, Eropa, dan Asia, dan umumnya tumbuh lebih baik di musim kemarau dibanding musim hujan. Karena itulah, anggur dikenal sebagai "tanaman hari panjang" atau Long Day Plant. Selain dimakan langsung, anggur juga sering diolah menjadi kismis, yang semakin memperkaya penggunaannya di dunia kuliner[[1](https://digitani.ipb.ac.id/mengenal-tanaman-anggur-morfologi-dan-karakteristiknya/)]. Dari sisi kesehatan, anggur menyimpan banyak manfaat. Beberapa di antaranya adalah membantu mencegah kanker, meningkatkan daya ingat, menjaga kesehatan mata, mengontrol tekanan darah, hingga memperlambat proses penuaan. Karena nilai manfaat dan ekonominya, kini semakin banyak masyarakat, terutama yang tinggal di daerah pesisir, mulai membudidayakan berbagai jenis anggur. Namun, masih banyak orang—terutama pedagang dan pecinta buah anggur—yang belum benar-benar memahami perbedaan jenis anggur berdasarkan warnanya. Selama ini, pengelompokan jenis anggur masih dilakukan secara manual hanya dengan mengandalkan penglihatan, sehingga sering terjadi perbedaan pendapat antar individu dalam mengidentifikasi jenis dan warnanya[[2](https://www.academia.edu/download/103003983/3016.pdf)]. Melihat tantangan tersebut, penulis menawarkan sebuah solusi berbasis teknologi, yaitu dengan menerapkan analisis prediktif dalam studi kasus klasifikasi kualitas buah anggur. Proses klasifikasi ini mempertimbangkan beberapa kriteria seperti tingkat kemanisan, keasaman, asal geografis, hingga intensitas paparan sinar matahari. Diharapkan, proyek ini dapat membantu para petani dalam menilai kualitas buah anggur secara otomatis, sehingga bisa meningkatkan efisiensi, produktivitas, serta nilai jual anggur di pasaran.

## Business Understanding
pengembangan model klasifikasi kualitas buah anggur memiliki potensi yang bermanfaat khususnya untuk para petani buah anggur. salah satu contoh potensi dan manfaat model klasifikasi ini adalah dapat membantu para petani melakukan pemilahan buah anggur secara otomatis sesuai dengan kriteria yang sudah diklasifikasikan oleh model.

### Problem Statements
Berdasarkan latar belakang di atas, berikut ini adalah rincian masalah yang dapat diselesaikan proyek ini:
- Jenis model apa yang memiliki akurasi terbaik untuk studi kasus klasifikasi kualitas buah anggur?
- Bagaimana model ini bisa membantu petani meningkatkan kualitas buah anggur mereka?

### Goals
Tujuan dari proyek ini antara lain adalah sebagai berikut:
- Membuat model machine learning yang dapat mengklasifikasikan kualitas buah anggur berdasarkan dataset yang terdiri dari data baris dan kolom.
- mengembangkan sebuah sistem atau aplikasi yang berbasis machine learning untuk membantu petani mengklasifikasikan kualitas buah anggur secara otomatis.
- Membandingkan beberapa algoritma model untuk menemukan model mana yang paling cocok dan memiliki akurasi terbaik untuk kasus klasifikasi kualitas buah anggur.

### Solution statements
- Melakukan Exploratory Data Analysis (EDA). Pada tahap ini, akan dilakukan analisis univariat dan multivariat, dilengkapi dengan visualisasi data untuk membantu memahami karakteristik data secara menyeluruh. EDA bertujuan untuk menemukan pola, tren, serta mengidentifikasi hubungan atau korelasi antar variabel. Temuan dari tahap ini akan menjadi dasar dalam pengambilan keputusan untuk pengembangan model machine learning yang lebih optimal.
- Dalam proyek ini, beberapa model machine learning dibuat untuk membandingkan dan menentukan model mana yang paling tepat digunakan dalam prediksi kualitas buah anggur. Beberapa metode yang digunakan antara lain:
    - Naive Bayes<br>
      Naive Bayes adalah metode klasifikasi berdasarkan Teorema Bayes dengan asumsi bahwa setiap fitur bersifat independen. Meskipun asumsi ini jarang terpenuhi sepenuhnya dalam kenyataan, model ini tetap efektif dan banyak digunakan, terutama untuk teks atau data kategorikal.
    - Support Vector Machine (SVM)<br>
      SVM adalah algoritma klasifikasi yang bekerja dengan mencari hyperplane terbaik yang memisahkan data ke dalam dua kelas secara optimal. Dalam implementasinya, SVM bisa digunakan untuk klasifikasi linear maupun non-linear menggunakan kernel trick.
    - K-Nearest Neighbor (KNN)<br>
      KNN adalah algoritma yang digunakan untuk klasifikasi dan regresi, dengan cara membandingkan data baru dengan sejumlah data terdekat di sekitarnya (tetangga terdekat). Model ini tidak membangun fungsi prediksi secara eksplisit, melainkan menyimpan semua data pelatihan dan mengklasifikasikan berdasarkan mayoritas label dari tetangganya.
    - Random Forest<br>
      Random Forest merupakan metode ensemble learning yang menggunakan kombinasi dari banyak decision tree. Setiap pohon akan memberikan prediksi, dan hasil akhirnya adalah kombinasi (rata-rata atau voting) dari semua prediksi tersebut. Ini membuat Random Forest tahan terhadap overfitting dan cocok untuk klasifikasi maupun regresi.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).
**Informasi tentang dataset**
| Kategori | Keterangan |
| Nama Dataset | Grape Quality |
| Sumber | [Kaggle.com](https://www.kaggle.com/datasets/mrmars1010/grape-quality) |
| Pemilik | [Mars_1010](https://www.kaggle.com/mrmars1010) |
| Lisensi | Apache 2.0 |
| Visibilitas | Publik |
| Tag | food |
| Skor Usibilitas | 10.00 |

### Variabel-variabel pada Grape Quality Dataset adalah sebagai berikut:
- sample_id: Nomor unik untuk setiap sampel anggur. Seperti nomor identitas untuk membedakan satu sampel dengan lainnya.
- variety: Jenis atau varietas anggur (contoh: Carbenet Sauvignon, Merlot, dll).
- region: Wilayah atau daerah tempat anggur ditanam. Setiap daerah punya iklim berbeda yang mempengaruhi kualitas anggur.
- quality_score: Skor angka yang menunjukkan sebebarapa bagus kualitas anggur (semakin tinggi, semakin bagus).
- quality_category: Kategori dari skor kualitas, dikelompokkan menjadi "Low", "Medium", "High", dan "Premium". Ini memudahkan untuk studi kasus klasifikasi.
- sugar_content_brix: Tingkat rasa manis anggur, diukur dalam satuan Brix (°Bx). Semakin tinggi Brix, semakin manis anggurnya.
- acidity_ph: Tingkat keasaman anggur. pH < 7 berarti asam. semakin rendah pH, semakin asam rasa anggurnya.
- cluster_weight_g: Berat satu tandan (sekumpulan) buah anggur, dalam satuan gram.
- harvest_date: Tanggal kapan anggur dipanen. Waktu panen bisa mempengaruhi rasa manis, asam, dan kualitas keseluruhan.
- sun_exposure_hours: Total jumlah jam anggur terkena sinar matahari. Sinar matahari membantu anggur matang dengan baik.
- soil_moisture_percent: Persentase kadar air dalam tempat anggur tumbuh. Ini mempengaruhi pertumbuhan dan rasa buah.
- rainfall_mm: Jumlah curah hujan (dalam milimeter) yang diterima kebun anggur selama masa tumbuh.

### Exploratory Data Analysis - Univariate Analysis

[Visualisasi Univariat (Data Kategori)](images/quality_category_img.png)
Gambar 1. Visualisasi Analisis Univariat (Data Kategori)

Pada gambar 1 di atas, dapat diketahui bahwa pada fitur quality_category kategori kualitas yang terbanyak adalah Medium, diikuti High dan untuk Premium dan Low sangat sedikit, jumlahnya jauh berbeda, yaitu kurang dari 100 data.

### Exploratory Data Analysis - Multivariate Analysis
[Visualisi Matriks Korelasi](images/confusion_matrix.png)
Gambar 2. Matriks Korelasi

Pada gambar 2 di atas, terlihat bahwa terdapat korelasi yang cukup besar pada quality score dengan sugar_content_brix sebesar 0.69, berry_size_mm sebesar 0.48 dan sun_exposure_hours sebesar 0.54.

## Data Preparation
Tahapan data preparation atau persiapan data merupakan proses penting sebelum membangun model machine learning. Tujuan utamanya adalah memastikan data dalam kondisi bersih, konsisten, dan sesuai untuk dianalisis, sehingga model yang dibangun dapat menghasilkan prediksi yang akurat dan andal. Berikut adalah tahapan data preparation yang dilakukan dalam proyek ini:

1. Data Loading<br>
   Pada tahap ini, data diimpor dan dimuat ke dalam lingkungan kerja (Google Colab notebook) menggunakan pustaka (library) seperti pandas. Proses ini penting sebagai langkah awal agar data dapat diakses dan diolah lebih lanjut dalam proyek.
2. Data Assessing<br>
    Tahap ini dilakukan untu mengetahui kondisi awal dari dataset. Beberapa pengecekan yang dilakukan meliputi:
    - Pengecekan nilai kosong<br>Untuk memastikan tidak ada data yang hilang yang dapat mengganggu proses pelatihan model.
    - Pengecekan duplikasi<br>Untuk menghindari adanya baris data ganda yang dapat membuat model bias.
    - Pemeriksaan outlier<br>Untuk melihat apakah ada data ekstrem yang dapat memengaruhi distribusi dan hasil prediksi.
    Hasil yang diperoleh dari tahapan ini:<br>
    Setelah dilakukan pemeriksaan, tidak ditemukan nilai kosong, duplikat, maupun outlier. Hal ini menunjukkan bahwa dataset yang digunakan sudah cuup bersih dan siap untuk tahap selanjutnya.
3. Data Cleaning & Transformation
    Setelah data dinyatakan bersih, dilakukan beberapa langkah lanjutan untuk mempersiapkan data agar optimal dalam pelatihan model:
    - Encoding Fitur Kategori:
      Proses ini dilakukan untuk mengubah fitur kategori menjadi format numerik. teknik encoding yang digunakan adalah one-hot-encoding. Teknik ini digunakan agar model dapat memproses data kategorical dengan lebih baik, karena model machine learning umumnya hanya dapat bekerja dengan data numerik. 
    - Split dataset:
      Dataset dibagi menjadi dua bagian, yaitu 80% untuk data latih dan 20% untuk uji. Hal ini bertujuan untuk mengevaluasi performa model pada data yang belum pernah dilihat sebelumnya.
    - Standardisasi Fitur<br>
      Fitur numerik distandardisasi agar memiliki rata-rata = 0 dan standar deviasi = 1 menggunakan teknik StandardScaler. Ini penting karena beberapa algoritma machine learning (seperti SVM dan KNN) sangat sensitif terhadapa skala data.

## Modeling
Pada proyek ini terdapat 4 algoritma yang akan digunakan, yaitu:
1. Naive Bayes<br>
   - Kelebihan:<br>
     - Proses pelatihan dan prediksi sangat cepat, cocok untuk dataset besar.
     - Mudah untuk dipahami dan diimplementasikan.
   - Kekurangan:<br>
     - Tidak fleksibel terhadap data numerik yang kompleks atau tidak terdistribusi dengan normal.
     - Asumsi independensi antar fitur sering kali tidak realistis dan dapat menurunkan akurasi.
2. Support Vector Machine (SVM)<br>
   - Kelebihan:<br>
     - Sangat efektif pada data berdimensi tinggi (banyak fitur).
     - Dapat menangani hubungan non-linear melalui berbagai jenis kernel. 
   - Kekurangan:<br>
     - Komputasi bisa sangat lambat untuk dataset besar.
     - Pemilihan kernel dan parameter tuning bisa cukup kompleks.
3. K-Nearest Neighbors (KNN)<br>
   - Kelebihan:<br>
     - termasuk algoritma yang sederhana dan mudah dipahami, cocok untuk pemula yang baru belajar machine learning.
     - Dapat digunakan untuk klasifikasi dan regresi.
   - Kekurangan:<br>
     - sangat sensitif terhadap skala data dan outlier; butuh normalisasi
     - Prediksi bisa lambat karena harus menghitung jarak ke semua data pelatihan.
4. Random Forest<br>
   - Kelebihan:<br>
     - Akurasi tinggi dan kuat terhadap overfitting dibandingkan decision tree tunggal.
     - Dapat digunakan untuk klasifikasi, regresi dan estimasi fitur penting.
   - Kekurangan:<br>
     - Model lebih kompleks dan besar, memakan lebih banyak memori dan waktu.
     - Interpretasi hasil lebih sulit dibandingkan model pohon tunggal.

## Evaluation
Pada tahap evaluasi model, digunakan metrik akurasi (accuracy) sebagai ukuran kinerja. akurasi dihitung dengan menentukan persentase jumlah prediksi yang tepat dibandingkan dengan total keseluruhan prediksi yang dilakukan. Rumus yang digunakan adalah:

$$\text{Akurasi} = \frac{\text{TP + TN}}{\text{TN + TP + FN + FP}} \times 100\%$$

Keterangan:
- TP (True Positve): Jumlah data yang benar-benar positif dan berhasil diprediksi dengan benar sebagai positif.
- TN (True Negative): Jumlah data negatif yang diprediksi dengan benar sebagai negatif.
- FP (False Positive): Jumlah data negatif yang diprediksi secara tidak benar sebagai positif (kesalahan tipe 1).
- FN (False Negative): Jumlah data positif yang diprediksi secara tidak benar sebagai negatif (kesalahan tipe 2).

Rumus ini memecah akurasi menjadi rasio antara data yang diklasifikasikan dengan benar (TP dan TN) dengan jumlah total data. Mengalikan dengan 100% mengubah rasio menjadi persentase.

Berikut hasil accuracy 4 buah model yang dikembangkan pada proyek ini:

| Model | Akurasi |
| ----- | ----- |
| Naive Bayes | 0.93 |
| SVM | 0.93 |
| KNN | 0.885 |
| Random Forest | 0.975 |

[Hasil Skor Akurasi](images/accuracy_scores.png)
Tabel 3. Hasil Akurasi

Berdasarkan hasil pengujian pada empat algoritma klasifikasi pada tabel 3 di atas, diketahui bahwa model dengan akurasi tertinggi adalah Random Forest dengan nilai akurasi sebesar 0.975 atau 9.75%. Disusul oleh Naive Bayes dan SVM yang masing-masign mencatatkan akurasi sebesar 93%, serta KNN dengan akurasi 88.5%.<br>
Melihat perbandingan tersebut, Random Forest menjadi pilihan terbaik untuk digunakan dalam proyek ini karena mampu memberikan performa prediksi yang paling akurat di antara model lainnya. Random Forest sendiri dikenal sebagai algoritma yang cukup andal karena menggabungkan banyak decision tree untuk menghasilkan prediksi yang stabil dan kuat terhadap overfitting.<br>
Meskipun Naive Bayes dan SVM juga menunjukkan performa yang baik, keduanya masih berada sedikit di bawah Random Forest. Sementara itu, KNN, walaupun sederhana dan mudah dipahami, memiliki akurasi paling rendah dari keempat model yang diuji.
<br>
Dengan mempertimbangkan faktor akurasi dan keandalan model, maka Random Forest dipilih sebagai model utama untuk memprediksi kualitas apel dalam studi ini.

## Referensi
1. Digitani IPB (2024). MENGENAL TANAMAN ANGGUR: MORFOLOGI DAN KARAKTERISTIKNYA
2. Ciputra, A., Rachmawanto, E. H., & 
Susanto, A. (2018). Klasifikasi 
Tingkat Kematangan Buah Apel 
Manalagi Dengan Algoritma Naive 
Bayes Dan Ekstraksi Fitur Citra 
Digital. Simetris: Jurnal Teknik 
Mesin, Elektro Dan Ilmu 
Komputer, 9(1), 465-472. 
