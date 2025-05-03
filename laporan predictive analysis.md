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
- Mengajukan 2 atau lebih solution statement. Misalnya, menggunakan dua atau lebih algoritma untuk mencapai solusi yang diinginkan atau melakukan improvement pada baseline model dengan hyperparameter tuning.
- Solusi yang diberikan harus dapat terukur dengan metrik evaluasi.

## Data Understanding
Paragraf awal bagian ini menjelaskan informasi mengenai data yang Anda gunakan dalam proyek. Sertakan juga sumber atau tautan untuk mengunduh dataset. Contoh: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Restaurant+%26+consumer+data).

Selanjutnya uraikanlah seluruh variabel atau fitur pada data. Sebagai contoh:  

### Variabel-variabel pada Restaurant UCI dataset adalah sebagai berikut:
- accepts : merupakan jenis pembayaran yang diterima pada restoran tertentu.
- cuisine : merupakan jenis masakan yang disajikan pada restoran.
- dst

**Rubrik/Kriteria Tambahan (Opsional)**:
- Melakukan beberapa tahapan yang diperlukan untuk memahami data, contohnya teknik visualisasi data atau exploratory data analysis.

## Data Preparation
Pada bagian ini Anda menerapkan dan menyebutkan teknik data preparation yang dilakukan. Teknik yang digunakan pada notebook dan laporan harus berurutan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan proses data preparation yang dilakukan
- Menjelaskan alasan mengapa diperlukan tahapan data preparation tersebut.

## Modeling
Tahapan ini membahas mengenai model machine learning yang digunakan untuk menyelesaikan permasalahan. Anda perlu menjelaskan tahapan dan parameter yang digunakan pada proses pemodelan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan kelebihan dan kekurangan dari setiap algoritma yang digunakan.
- Jika menggunakan satu algoritma pada solution statement, lakukan proses improvement terhadap model dengan hyperparameter tuning. **Jelaskan proses improvement yang dilakukan**.
- Jika menggunakan dua atau lebih algoritma pada solution statement, maka pilih model terbaik sebagai solusi. **Jelaskan mengapa memilih model tersebut sebagai model terbaik**.

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

Pada bagian ini anda perlu menyebutkan metrik evaluasi yang digunakan. Lalu anda perlu menjelaskan hasil proyek berdasarkan metrik evaluasi yang digunakan.

Sebagai contoh, Anda memiih kasus klasifikasi dan menggunakan metrik **akurasi, precision, recall, dan F1 score**. Jelaskan mengenai beberapa hal berikut:
- Penjelasan mengenai metrik yang digunakan
- Menjelaskan hasil proyek berdasarkan metrik evaluasi

Ingatlah, metrik evaluasi yang digunakan harus sesuai dengan konteks data, problem statement, dan solusi yang diinginkan.

**Rubrik/Kriteria Tambahan (Opsional)**: 
- Menjelaskan formula metrik dan bagaimana metrik tersebut bekerja.

**---Ini adalah bagian akhir laporan---**

_Catatan:_
- _Anda dapat menambahkan gambar, kode, atau tabel ke dalam laporan jika diperlukan. Temukan caranya pada contoh dokumen markdown di situs editor [Dillinger](https://dillinger.io/), [Github Guides: Mastering markdown](https://guides.github.com/features/mastering-markdown/), atau sumber lain di internet. Semangat!_
- Jika terdapat penjelasan yang harus menyertakan code snippet, tuliskan dengan sewajarnya. Tidak perlu menuliskan keseluruhan kode project, cukup bagian yang ingin dijelaskan saja.

