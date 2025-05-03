"""
# Proyek Predictive Analytics: Orange Quality Data
- **Nama:** Rachmat Risky Saputra
- **Email:** rachmatrisky5@gmail.com
- **ID Dicoding:** rahmtris
"""

"""
## Deskripsi Proyek
"""

"""
### Latar Belakang Proyek Prediksi Kualitas Buah Anggur dengan Menggunakan Machine Learning
"""

"""
Proyek ini bertujuan untuk membangun model machine learning yang mampu memprediksi kualitas buah anggur dengan lebih akurat dan efisien. Saat ini, proses penilaian kualitas anggur masih dilakukan secara manual, yang membutuhkan banyak waktu, tenaga serta memiliki risiko kesalahan yang tinggi, contohnya jika buah anggur yang harus dinilai sangat banyak dan orang yang bertugas hanya sedikit, tentu dengan seiring berjalannya waktu, penilai-penilai ini akan kelelahan yang bisa mengurangi fokus mereka dalam menilai kualitas buah anggur. Kondisi ini dapat menyebabkan kerugian bagi petani, serta menghasilkan produk yang tidak memenuhi ekspetasi konsumen. Dengan adanya model prediksi kualitas buah anggur, harapannya masalah ini dapat diatasi melalui solusi yang lebih akurat dan efisien.
"""

"""
## Import library yang Diperlukan
"""

from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import BernoulliNB
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score
from google.colab import files

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import zipfile

"""
## Data Understanding
"""

"""
Data Understanding adalah tahap awal dalam proses analisis data yang bertujuan untuk memahami isi, struktur, dan karakteristik data yang tersedia. Pada tahap ini, dilakukan beberapa aktivitas utama seperti:
- Mengevaluasi kualitas data, termasuk mengidentifikasi data yang hilang, tidak konsisten, atau outlier.
- Memahami makna dari setiap fitur (kolom) dalam dataset.
- Menganalisis distribusi data untuk menemukan pola, tren, dan anomali.
- Menilai keterkaitan antar fitur yang dapat mempengaruhi performa yang dihasilkan oleh model.
"""

"""
### Data Loading
"""

"""
Data Loading adalah tahapan untuk memuat dataset yang akan digunakan ke dalam notebook.
"""

files.upload()

!mkdir ~/.kaggle
!cp kaggle.json ~/.kaggle
!chmod 600 ~/.kaggle/kaggle.json

!kaggle datasets download -d mrmars1010/grape-quality

zip_ref = zipfile.ZipFile("/content/grape-quality.zip", "r")
zip_ref.extractall("/content")
zip_ref.close()

grapes_df = pd.read_csv("GRAPE_QUALITY.csv")
grapes_df

"""
Output kode di atas memberikan informasi sebagai berikut:
- ada 1.000 baris (records atau jumlah pengamatan) dalam dataset.
- Terdapat 13 kolom yaitu: sample_id, variety, region, quality_score, quality_category, sugar_content_brix, acidity_ph, cluster_weight_g, berry_size_mm, harvest_date, sun_exposure_hours, soil_moisture_percent, rainfall_mm
"""

"""
## Exploratory Data Analysis (EDA)
"""

"""
### Exploratory Data Analysis - Deskripsi Variabel
"""

"""
Berdasarkan informasi dari kaggle, variabel-variabel pada dataset Grape Quality adalah sebagai berikut:
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
"""

grapes_df.info()

"""
Dari output terlihat bahwa:
- Terdapat 4 kolom dengan tipe object, yaitu: variety, region, quality_category dan harvest_date. Kolom ini merupakan categorical features (fitur non-numerik).
- Terdapat 8 kolom numerik dengan tipe data float64 yaitu: quality_score, sugar_content_brix, acidity_ph, cluster_weight_g, berry_size_mm, sun_exposure_hours, soil_moisture_percent, rainfall_mm.
- Terdapat 1 kolom numerik dengan tipe data int64, yaitu: sample_id.
"""

grapes_df.describe()

"""
Fungsi describe() memberikan informasi statistik pada masing-masing kolom, antara lain:
- Count adalah jumlah sampel pada data.
- Mean adalah nilai rata-rata.
- Std adalah standar deviasi.
- Min yaitu nilai minimum setiap kolom.
- 25% adalah kuartil pertama. Kuartil adalah nilai yang menandai batas interval empat bagian sebaran yang sama.
- 50% adalah kuartil kedua, atau biasa juga disebut median (nilai tengah).
- 75% adalah kuartil ketiga.
- Max adalah nilai maksimum
"""

grapes_df.shape

"""
### Explatory Data Analysis - Menangani Missing Value, Duplicated Value dan Outliers
"""

grapes_df.isnull().sum()

"""
Dari output di atas, terlihat bahwa tidak ada missing values pada dataset.
"""

grapes_df.duplicated().sum()

"""
Dari output di atas, terlihat bahwa tidak ada data duplikat pada dataset.
"""

outliers = grapes_df.select_dtypes(exclude=["object"])
for column in outliers:
    plt.figure()
    sns.boxplot(data=outliers, x=column)

"""
Berdasarkan output di atas, bisa diketahui bahwa tidak ada outliers dalam dataset ini, jadi kita tidak perlu melakukan penanganan outliers.
"""

"""
### Exploratory Data Analysis - Univariate Analysis
"""

numerical_features = ["quality_score", "sugar_content_brix", "acidity_ph", "cluster_weight_g", "berry_size_mm", "sun_exposure_hours", "soil_moisture_percent", "rainfall_mm"]
categorical_features = ["variety", "region", "harvest_date", "quality_category"]

"""
#### Fitur Kategori
"""

feature = categorical_features[0]
count = grapes_df[feature].value_counts()
percentage = 100 * grapes_df[feature].value_counts(normalize=True)
df = pd.DataFrame({"Jumlah sampel": count, "persentase": percentage.round(1)})
print(df)
count.plot(kind="bar", title=feature)

feature = categorical_features[1]
count = grapes_df[feature].value_counts()
percentage = 100 * grapes_df[feature].value_counts(normalize=True)
df = pd.DataFrame({"Jumlah sampel": count, "persentase": percentage.round(1)})
print(df)
count.plot(kind="bar", title=feature)

feature = categorical_features[2]
count = grapes_df[feature].value_counts()
percentage = 100 * grapes_df[feature].value_counts(normalize=True)
df = pd.DataFrame({"Jumlah sampel": count, "persentase": percentage.round(1)})
print(df)

plt.figure(figsize=(15, 5))
count.plot(kind="bar", title=feature)
plt.xticks(rotation=45, ha="right")
plt.tight_layout()
plt.show()

feature = categorical_features[3]
count = grapes_df[feature].value_counts()
percentage = 100 * grapes_df[feature].value_counts(normalize=True)
df = pd.DataFrame({"Jumlah sampel": count, "persentase": percentage.round(1)})
print(df)
count.plot(kind="bar", title=feature)

"""
#### Fitur Numerik
"""

grapes_df.hist(bins=50, figsize=(20, 15))
plt.show()

"""
### Exploratory Data Analysis - Multivariate Analysis
"""

"""
#### Fitur Numerik
"""

sns.pairplot(grapes_df, diag_kind="kde")

plt.figure(figsize=(10, 8))
corr_matrix = grapes_df[numerical_features].corr()

sns.heatmap(data=corr_matrix, annot=True, cmap="coolwarm", linewidths=0.5)
plt.title(f"Matriks korelasi untuk fitur numerik", size=20)

"""
## Data Preparation
"""

"""
Data Preparation adalah adalah tahapan untuk mempersiapkan dataset melalui proses seperti data cleaning, data structuring, data transformation dll. Agar dataset siap digunakan untuk pelatihan model machine learning.
"""

"""
### Data Cleaning
"""

"""
### Encoding Fitur Kategori
"""

categorical_features

grapes_df = pd.concat([grapes_df, pd.get_dummies(grapes_df["variety"], prefix="variety")], axis=1)
grapes_df = pd.concat([grapes_df, pd.get_dummies(grapes_df["region"], prefix="region")], axis=1)
grapes_df.drop(["variety", "region"], axis=1, inplace=True)
grapes_df.head()

custom_mapping = {
    "Low": 0,
    "Medium": 1,
    "High": 2,
    "Premium": 3,
}

grapes_df["quality_category_encoded"] = grapes_df["quality_category"].map(custom_mapping)
print(grapes_df[["quality_category", "quality_category_encoded"]].head())

grapes_df.drop(["quality_category"], axis=1, inplace=True)

grapes_df.drop(["sample_id"], axis=1, inplace=True)

grapes_df.drop(["harvest_date"], axis=1, inplace=True)

"""
### Train-Test-Split
"""

X = grapes_df.drop(["quality_category_encoded"], axis=1)
y = grapes_df[["quality_category_encoded"]]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

"""
### Standarisasi data training
"""

scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

"""
## Model Development
"""

models = pd.DataFrame(index=["accuracy_score"],
                        columns=["Naive Bayes", "SVM", "KNN", "Random Forest",])

"""
### Pelatihan model Naive Bayes
"""

nb_model = BernoulliNB()
nb_model.fit(X_train, y_train)

"""
### Pelatihan model SVC
"""

svc_model = SVC()
svc_model.fit(X_train, y_train)

"""
### Pelatihan model KNN
"""

KNN_model = KNeighborsClassifier(n_neighbors=5, weights="distance")
KNN_model.fit(X_train, y_train)

"""
### Pelatihan model Random Forest
"""

rf_model = RandomForestClassifier(max_depth=20)
rf_model.fit(X_train, y_train)

"""
## Evaluasi Model
"""

"""
### Standarisasi data testing
"""

X_test[numerical_features] = scaler.transform(X_test.loc[:, numerical_features])
X_test[numerical_features].head()

"""
### Melakukan testing model dengan data tes
"""

nb_prediction = nb_model.predict(X_test)
models.loc["accuracy_score", "Naive Bayes"] = accuracy_score(y_test, nb_prediction)

svc_prediction = svc_model.predict(X_test)
models.loc["accuracy_score", "SVM"] = accuracy_score(y_test, svc_prediction)

KNN_prediction = KNN_model.predict(X_test)
models.loc["accuracy_score", "KNN"] = accuracy_score(y_test, KNN_prediction)

rf_prediction = rf_model.predict(X_test)
models.loc["accuracy_score", "Random Forest"] = accuracy_score(y_test, rf_prediction)

"""
### Hasil akurasi model dalam dataframe
"""

models

"""
### Plot visualisasi akurasi model
"""

plt.bar('Naive Bayes', models['Naive Bayes'])
plt.bar('SVC', models['SVM'])
plt.bar('KNN', models['KNN'])
plt.bar('Random Forest', models['Random Forest'])
plt.title("Hasil Akurasi Model");
plt.xlabel('Model');
plt.ylabel('Akurasi');
plt.show()