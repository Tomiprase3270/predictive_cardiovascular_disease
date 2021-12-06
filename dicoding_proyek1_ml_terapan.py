# -*- coding: utf-8 -*-
"""Dicoding_proyek1_ML_Terapan.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WaKRUU9kvEjbtvYaO9N5BG5fgaHyk0Bq

# Cardiovascular Disease classification for health company prediction - by Tomi Prasetyo

## Domain Proyek

Menurut data dari WHO (World Health Organization), kardiovaskular atau penyakit yang berhubungan dengan jantung dan pembuluh darah merupakan penyebab kematian terbanyak di seluruh dunia. Pada tahun 2019 ada sekitar 17,9 juta orang meninggal karena kardiovaskular(https://www.who.int/news-room/fact-sheets/detail/cardiovascular-diseases-(cvds)). Sedangkan di Indonesia sendiri, menurut data dari Riset Kesehatan Dasar(Riskesdas) yang di kutip dari kementerian kesehatan, Setidaknya, 15 dari 1000 orang, atau sekitar 2.784.064 individu di Indonesia menderita penyakit jantung(http://p2ptm.kemkes.go.id/kegiatan-p2ptm/pusat-/hari-jantung-sedunia-world-heart-day-your-heart-is-our-heart-too).
Berdasarkan data tersebut tentu diperlukan deteksi dini untuk menangantisipasi peningkatan penyakit jantung pada pasien, yang nantinya dapat digunakan oleh perusahaan yang bergerak di bidang kesehatan maupun rumah sakit, sehingga bisa dilakukan tindakan pencegahan dan perawatan.

## Bussiness Understanding

Tentu saat ini rumah sakit sudah dilengkapi dengan alat yang dapat untuk merekam data berbagai kondisi dari pasien.
Dengan data yang banyak dari berbagai rumah sakit maupun institusi yang tersedia di internet dapat digunakan untuk menyusun algoritma machine learning dalam hal klasifikasi apakah kemungkinan seorang pasien mempunyai peluang untuk memiliki masalah jantung dan pembuluh darah.

Dengan sebuah sistem prediksi yang akurat, dapat meningkatkan kepercayaan pasien terhadap perusahaan yang bergerak di bidang kesehatan tersebut. Semakin banyak pasien yang mempercayakan kesehatannya kepada perusahaan tersebut, maka income pada perusahaan tersebut akan semakin besar pula.

### Problem Statement

Berdasarkan penjelasan diatas, perusahaan yang bergerak di bidang kesehatan dapat membuat sebuah sistem prediksi untuk mengklasifikasikan kemungkinan seorang pasien menderita kardiovaskular, dan bisa menjawab permasalahan berikut :

- dari semua fitur yang ada fitur apa saja yang punya faktor terbesar sesorang mengidap kardiovaskular?
- Apakah seseorang dengan data kesehatan tertentu diprediksi mempunyai resiko kardiovaskular?

### Goals

Untuk menjawab pertanyaan tersebut, saya akan membuat klasifikasi modelling dengan tujuan atau goals sebagai berikut:

- mengetahui fitur yang berrelasi terhadap klasifikasi kemungkinan kardiovaskular atau tidak.
- Membuat model machine learning yang dapat mengklasifikasikan apakah sesorang kemungkinan dapat mengidap kardiovaskular dengan seakurat mungkin berdasarkan fitur-fitur yang ada.

### Solution Statement

Pada kasus ini, saya akan membuat sebuah sistem prediksi untuk kalsifikasi dengan memnfaatkan algoritma machine learning Random Forest.

Random Forest adalah termasuk kedalam kelompok ensemble, dimana didalamnya terdapat bebrapa kelompok model machine learning sederhana yang secara bersama-sama menghasilkan model yang lebih powerfull.

- model yang akan saya pakai adalah Random Forest Classifier karena kita akan melakukan proses klasifikasi. Model ini pada dasarkan terdiri dari beberapa model Decision Tree(Pohon Keputusan), dimana keluaran dari model diambil dari keluaran mana yang terbanyak.
- Pada model saya mengadopsi teknik Bagging, dimana setiap model Decision Tree akan mengambil fitur secara acak dan setiap model akan menghasilkan keluaran yang berbeda.

## Data Understanding

Data yang akan saya gunakan dalam proyek ini merupakan data yang didapat dari berbagai pengecakan medis dan tersedia secara bebas di situs dataset kaggle (https://www.kaggle.com/sulianova/cardiovascular-disease-dataset)

Dataset ini terdiri dari 70.000 data dengan 11 feature dan 1 label sebagai berikut:

### Load dataset
Setelah dataset kita unduh pada website diatas, kita dapat menyimpannya pada google drive.
Kemudian kita dapat menghubungkan colab dengan Google drive melalui kode dibawah
"""

from google.colab import drive
drive.mount('/content/drive')

data = "/content/drive/MyDrive/Colab Notebooks/dataset/cardio_train.csv"

"""### Feature

terdiri dari dua jenis data, yaitu data numerik dan data kategori.

data numerik meliputi : id, age, height, weight, ap_hi dan ap_lo,

sedangkan data kategori meliputi : gender, cholesterol, gluc, alco, active

Berikut penjelasan detail dari feature pada dataset

- id : nomor urut pasien
- age : merupakan umur pasien (dalam satuan hari)
- height : merupakan tinggi pasien (dalam satuan sentimeter)
- weight : merupakan berat badan pasien (dalam satuan kilogram)
- Gender : merupakan jenis kelamin pasien
- ap_hi : merupakan tekanan darah sistolik (dalam satuan mmHg)
- ap_lo : merupakan tekanan darah diastolik (dalam satuan mmHg)
- cholesterol : merupakan kadar kolesterol dalam kategori, yaitu normal, diatas normal dan jauh diatas normal
- gluc : merupakan kadar glucosa dalam kategori, yaitu normal, diatas normal dan jauh diatas normal
- smoke : merupakan data kategori apakah pasien merokok atau tidak
- alco : merupakan data kategori apakah pasien sering mengkonsumsi alkohol atau tidak
- active : merupakan data kategori apakah pasien sering bergerak atau tidak

### Label

Label disini merupakan satu kolom dengan nama kolom "cardio".

Terdiri dari data kategori 0 dan 1, dimana 0 adalah "no cardio (kemungkinan tidak kardiovaskular)" dan 1 adalah "cardio (kemungkinan akan mempunyai kardiovaskular)"

import  Library yang diperlukan. 
Disini saya memakai pandas dan numpy untuk mengolah dataframe, serta matplotlib, seaborn untuk visualisasi.
"""

# Commented out IPython magic to ensure Python compatibility.
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# %matplotlib inline
import seaborn as sns

df = pd.read_csv(data, sep=";")
df.head()

"""## Data Preparation

membaca file apakah ada value missing value
"""

df.info()

"""Menghilangkan kolom id, karena kolom id tidak ada korelasi dengan proses klasifikasi"""

df.drop(columns=['id'], inplace=True)
df.head()

"""Merubah value dari atribut "age" menjadi format tahun(jika pada dataframe sebelumnya adalah hari)"""

df['age'] = round(df['age'] /365)
df.head()

"""

melakukan pengecekan terhadap data duplikat
"""

df.duplicated().sum()

"""menghapus data yang terduplikasi"""

df.drop_duplicates(inplace=True)
df

"""melihat sebaran data pada target"""

cardio = df['cardio'].value_counts()
cardio.plot(kind='bar', title='cardio')

"""melakukan pengecekan desripsi data dengan fungsi describe()"""

df.describe()

"""Dari data tersebut kita mendapati uraian sebagai berikut :


---



*   minimum umur adalah 30 tahun dan maksimum adalah 65 tahun
*   minimum tinggi adalah 55 cm dan maksimum 250 cm
*   minimum berat adalah 10 kg dan maksimum berat adalah 200 kg
*   minimumm tekanan darah sistolik adalah -150 mmHg dan maksimum 16.020 mmHg
*   minimum tekanan darah diastolik adalah -70 mmHg dan maksimum 11.000 mmHg


dari data tersebut dapat kita simpulkan bahwa ada outlier data, dimana pada tinggi 55 cm dengan usia minimal 30 tahun terlihat begitu rancu, apalagi dengan tekanan darah minimum di -150 dan maksimum 16.020 tentu akan menjadi pertimbangan dari data tersebut.

### Exploratory Data Analysis

outlier data

menghapus data ambigu. yaitu data yang mempunyai rentang terlalu jauh dengan rata-rata data. Ataupun bisa kita katakan data yang tidak sesuai pada kenyataan. Seperti tekanan darah yang terlalu tinggi, ataupun jauh dari batas rendah, tentu itu tidak terjadi di dunia nyata.
"""

sns.boxplot(x=df['height'])
plt.title('distribution of Height')

sns.boxplot(x=df['weight'])
plt.title('distribution of Weight')

sns.boxplot(x=df['ap_hi'])
plt.title('distribution of Systolic blood pressure')

sns.boxplot(x=df['ap_lo'])
plt.title('distribution of Dyastolic blood pressure')

min_height = (df.height <= 140)
print('jumlah tinggi kurang dari 140 cm = ', min_height.sum(), 'data')
max_height = (df.height >= 200)
print('jumlah tinggi lebih dari 200 cm = ', max_height.sum(), 'data')
min_weight = (df.weight <= 35)
print('jumlah berat kurang dari 35 kg = ', min_weight.sum(), 'data')
max_weight = (df.weight >= 120)
print('jumlah berat lebih dari 120 kg = ', max_weight.sum(), 'data')
max_ap_hi = (df.ap_hi >= 300)
print('tekanan darah sistolik lebih dari 300 mmHg = ', max_ap_hi.sum(), 'data')
min_ap_hi = (df.ap_hi <= 50)
print('tekanan darah sistolik kurang dari 50 mmHg = ', min_ap_hi.sum(), 'data')
max_ap_lo = (df.ap_lo >= 250)
print('tekanan darah diastolik lebih dari 250 mmHg = ', max_ap_lo.sum(), 'data')
min_ap_lo = (df.ap_lo <= 40)
print('tekanan darah diastolik kurang dari 40 mmHg = ', min_ap_lo.sum(), 'data')

"""Menghapus data outlier"""

dataset_outlier = df.loc[(min_height) | (max_height) | (min_weight) | (max_weight) | (max_ap_hi) | (min_ap_hi) | (max_ap_lo) | (min_ap_lo)]
dataset_outlier

"""sekitar 2020 data outlier yang berhasil kita kumpulkan.


Selanjutnya kita akan melihat besaran label pada outlier data kita
"""

cardio_outlier = dataset_outlier['cardio'].value_counts()
cardio_outlier.plot(kind='bar', title='cardio')

cardio_outlier

"""menghapus data outlier. kita disini bisa memkai fungsi drop."""

df = df.drop(dataset_outlier.index)
df

sns.boxplot(x=df['height'])
plt.title('distribution of Height after cleaning')

sns.boxplot(x=df['weight'])
plt.title('distribution of Weight after cleaning')

"""Cek jumlah target"""

sns.boxplot(x=df['ap_hi'])
plt.title('distribution of Systolic after cleaning')

sns.boxplot(x=df['ap_lo'])
plt.title('distribution of Dyastolic after cleaning')

cardio = df['cardio'].value_counts()
cardio.plot(kind='bar', title='cardio')
cardio

"""dataset sudah kita bersikan dari data yang ambigu

### Univariate Analysis

Membagi data menjadi data numerik dan data kategorical
"""

numerical_features = ['age','height', 'weight', 'ap_hi', 'ap_lo']
categorical_features = ['gender', 'cholesterol', 'gluc', 'smoke', 'alco', 'active', 'cardio']

"""Categorical Feature

melakukan analysis pada data kategori dengan visualisasi data

*   gender

membandingkan pengidap Cardiovascular berdasarkan gender
"""

pd.crosstab(df['cardio'], df['gender']).plot(kind = 'bar')
plt.xlabel('0 = no heart disease, 1 = heart disease')
plt.legend(['woman','man'])
plt.show()

feature = categorical_features[0]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df_new = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_new)
count.plot(kind='bar', title=feature);

"""Dilihat dari data bahwa 64,4% dari dataset berjenis kelamin perempuan dan 35 % laki-laki

dengan jumlah penderita CVD lebih banyak perempuan

*   cholesterol
"""

pd.crosstab(df['cardio'], df['cholesterol']).plot(kind = 'bar', figsize=(9,6))
plt.xlabel('1: normal, 2: above normal, 3: well above normal')
plt.legend(['normal','above normal', 'well above normal'])
plt.show()

feature = categorical_features[1]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df_new = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_new)
count.plot(kind='bar', title=feature);

"""pada kolom cholesterol, dapat dilihat bahwa 73,8 % tidak mempunyai masalah pada kolesterol, tetapi pada penderita CVD dibarengi dengan peningkatan pada jumlah pengidap kolesterol.

*   gluc
"""

pd.crosstab(df['cardio'], df['gluc']).plot(kind = 'bar', figsize=(9,6))
plt.xlabel('1: normal, 2: above normal, 3: well above normal')
plt.legend(['normal','above normal', 'well above normal'])
plt.show()

feature = categorical_features[2]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df_new = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_new)
count.plot(kind='bar', title=feature);

"""Data berupa 84,4% dengan penderita kadar glucosa yang normal

tetapi pada data menunjukkan pada penderita CVD, ada peningkatan glucosa

*   smoke
"""

pd.crosstab(df['cardio'], df['smoke']).plot(kind = 'bar', figsize=(9,6))
plt.xlabel('0: no smoke, 1: smoke')
plt.legend(['no smoke','smoke'])
plt.show()

feature = categorical_features[3]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df_new = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_new)
count.plot(kind='bar', title=feature);

"""data sampel menunjukkan 90,7% tidak merokok, dan merokok tidak berkontribusi pada data yang signifikan

*   alco
"""

pd.crosstab(df['cardio'], df['alco']).plot(kind = 'bar', figsize=(9,6))
plt.xlabel('0: no alcohol, 1: alcohol')
plt.legend(['no alcohol','alcohol'])
plt.show()

feature = categorical_features[4]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df_new = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_new)
count.plot(kind='bar', title=feature);

"""data 94,4% tidak minum alkohol, dan kurang dari 300 orang penderita CVD akibat mengkonsumsi alkohol

*   active
"""

pd.crosstab(df['cardio'], df['active']).plot(kind = 'bar', figsize=(9,6))
plt.xlabel('0: no active, 1: active')
plt.legend(['no active','active'])
plt.show()

feature = categorical_features[5]
count = df[feature].value_counts()
percent = 100*df[feature].value_counts(normalize=True)
df_new = pd.DataFrame({'jumlah sampel':count, 'persentase':percent.round(1)})
print(df_new)
count.plot(kind='bar', title=feature);

"""dapat disimpulkan bahwa punya kegiatan yang aktif tidak terlalu banyak mempengaruhi sesorang terkena serangan jantung

Berdasarkan visualisasi data tersebut diatas dapat disimpulkan bahwa kenaikan kolesterol dan glukosa dalam darah memberikan efek pada kemungkinan terjadi serangan jantung pada pasien

>Selanjutnya kita akan observasi penyakit jantung berdasarkan umur, tinggi berat, dan tekanan darah
"""

sns.violinplot(x=df.cardio, y= df.age)
print("Observations cardio by age");

sns.violinplot(x=df.cardio, y= df.height)
print("Observations cardio by height");

sns.violinplot(x=df.cardio, y= df.weight)
print("Observations cardio by weight");

sns.violinplot(x=df.cardio, y= df.ap_hi)
print("Observations cardio by Systolic blood pressure");

sns.violinplot(x=df.cardio, y= df.ap_lo)
print("Observations cardio by Dyastolic blood pressure");

"""Dari visualisasi tersebut dapat kita simpulkan bahwa penyakit jantung lebih banyak terjadi di umur lebi dari 50 tahun.

Demikian juga pada tekanan darah yang lebih tinggi dapat membuat kemungkinan seseorang terkena serangan jantung lebih besar.

### Multivariate Analysis

Corelation Score
"""

corr = df.corr()
cmap = sns.diverging_palette(220, 10, as_cmap=True)

mask = np.zeros_like(corr, dtype=np.bool)
mask[np.triu_indices_from(mask)] = True

f, ax = plt.subplots(figsize=(11, 9))
sns.heatmap(corr, mask=mask, cmap=cmap, vmax=.3, center=0,annot = True,
            square=True, linewidths=.5, cbar_kws={"shrink": .5});

"""dari corelation metric tersebut dapat kita ketahui bahwa yang paling berperan menyebabkan sesorang terkena serangan jantung adalah umur, tekanan darah, kolesterol, dan berat badan.

### Data Preocessing

Setelah kita melihat dan mengamati persebaran data, kita akan melakukan penyetaraan data, yaitu proses untuk membuat nilai pada tiap kolom mempunyai nilai dalam rentang yang sama, sehingga model kita akan mudah melakukan proses training.
"""

df.head()

"""Mengubah gender dari "1:women, 2:men", menjadi "0:women, 1:men""""

df['gender'] = df.gender.replace([1,2], [0,1])
df.head()

"""Melakukan proses encoding pada data tipe kategori. Ini bertujuan untuk mempermudah model dalam memproses data.
  Sebelumnya pada data kategori, yaitu cholesterol, gluc, smoke, alco, dan active terdiri dari dua maupun tiga kategori dalam satu kolom. dengan encoding kita akan membuat masing masing kategori menjadi kolom baru, dan kolom yang memuat kategori tersebut akan bernilai 1, sedangkan yang tidak akan bernilai 0.
  Dengan encoding diharapkan nantinya akan memudahkan model untuk melakukan proses training dan evaluasi.
  pada scikit learn tersedia fitur OneHotEncoder yang berguna untuk melakukan proses encoding.

one hot encoding data categorical yang memiliki 3 value, yaitu cholesterol dan gluc
"""

df['cholesterol']=df['cholesterol'].map({ 1: 'normal', 2: 'above normal', 3: 'well above normal'})
df['gluc']=df['gluc'].map({ 1: 'normal', 2: 'above normal', 3: 'well above normal'})
encoding_features = pd.get_dummies(df[['cholesterol','gluc']],drop_first=True)
df = pd.concat([df,encoding_features],axis=1)
df.drop(['cholesterol','gluc'],axis=1,inplace=True)
df.head()

"""### Split Data

membagi data menjadi dua jenis yaitu data train dan data test.
data train akan saya gunakan pada tahap training, dan data test akan saya gunakan pada tahap evaluasi.
Proses membagi data menggunakan fungsi train_test_split dari library scikit learn.
"""

from sklearn.model_selection import train_test_split
X = df.drop(["cardio"],axis =1)
y = df["cardio"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 56)

"""Cek proporsi data"""

print(f'Total of dataset: {len(X)}')
print(f'total off train dataset: {len(X_train)}')
print(f'Total of  test dataset: {len(X_test)}')

"""Melihat perbandingan label "cardio" dan "no cardio" pada y_train dan y_test dengan matplotlib"""

label_train = y_train.value_counts()
label_test = y_test.value_counts()
plt.figure(0)
label_train.plot(kind='bar')
plt.figure(1)
label_test.plot(kind='bar')

"""### Standarisasi

Setelah melakukan proses encoding pada setiap data kategori maka akan membentuk nilai 1 dan 0 pada masing masing kategori, selanjutnya saya melakukan proses standarisasi pada data numerik.
  Standardisasi adalah teknik transformasi yang paling umum digunakan dalam tahap persiapan pemodelan. Untuk data numerik, kita tidak akan melakukan transformasi dengan one-hot-encoding seperti pada fitur kategori. Kita akan menggunakan teknik StandarScaler dari library Scikitlearn.
  StandardScaler melakukan proses standarisasi fitur dengan mengurangkan mean (nilai rata-rata) kemudian membaginya dengan standar deviasi untuk menggeser distribusi. StandardScaler menghasilkan distribusi dengan standar deviasi sama dengan 1 dan mean sama dengan 0. Sekitar 68% dari nilai akan berada di antara -1 dan 1.

"""

from sklearn.preprocessing import StandardScaler
numerical_features = ['age', 'height', 'weight', 'ap_hi', 'ap_lo']
scaler = StandardScaler()
scaler.fit(X_train[numerical_features])
X_train[numerical_features] = scaler.transform(X_train.loc[:, numerical_features])
X_train[numerical_features].head()

X_train

"""cek keseimbangan jumlah target """

y_train.value_counts()

"""# Prediksi

Karena disini adalah mengklasifikasikan apakah pasien ada kemungkinan mengidap kardiovaskular maka saya menggunakan algoritma klasifikasi dari Random Forest Classifier dari scikit learn.
Random Forest bisa dipakai untuk klasifikasi maupun regresi. Random Forest termasuk kedalam kelompok ensemble(group). model ensemble adalah kelompok model yang terdiri dari beberapa model yang bekerja secara bersama-sama. Dari setiap model yang ada didalam kelompok ensemble ini akan membuat prediksi secara independen sehingga prediksi dari satu model dengan model yang lain tentu akan berbeda. Prediksi dari setiap model ini akan digabungkan untuk menjadi prediski akhir model nesemble.

ada 2 pendekatan dalam model ensemble, yaitu bagging dan boosting. teknik bagging sangat cocok untuk model Decision Tree. Karena sejatinya Random Forest tersusun atas kumpuan dari model Decision Tree maka, model ini akan memakai teknik bagging. Bagging atau bootstrap aggregating adalah teknik yang melatih model dengan sampel random. Dalam teknik bagging, sejumlah model dilatih dengan teknik sampling with replacement (proses sampling dengan penggantian). Ketika kita melakukan sampling with replacement, sampel dengan nilai yang berbeda bersifat independen. Artinya, nilai suatu sampel tidak mempengaruhi sampel lainnya. Akibatnya, model yang dilatih akan berbeda antara satu dan lainnya.
Hasil prediksi akhir adalah jumlah terbanyak yang diprediksi oleh kumpulan model Decision Tree.

Pada tahap training saya mendapatkan akurasi, saya menggunakan metrik accurasi score dari scikit learn dan didapatkan akurasi sebesar 0,82, dan untuk menghitung loss saya menggunakan metrik mean squared error dan didapatkan error sebesar 0,17.

Mengenai metrics acurracy_score dan mean_squared_error akan saya jelaskan pada tahap evaluasi.
"""

# Impor library yang dibutuhkan
from sklearn.metrics import mean_squared_error, accuracy_score
from sklearn.ensemble import RandomForestClassifier

# buat model prediksi
clf = RandomForestClassifier(n_estimators=300, max_depth=16, random_state=55, n_jobs=-1)
clf.fit(X_train, y_train)
acc = accuracy_score(clf.predict(X_train), y_train)
print('score : ', acc)
error = mean_squared_error(y_true=y_train, y_pred=clf.predict(X_train))
print('error : ', error)

"""# Evaluasi


Pada tahap evaluasi saya memakai data test yang berbeda dengan data training.

metrik evaluasi yang digunakan juga sama seperti pada tahap training yaitu memakai metrik accurasi score dan untuk menghitung loss menggunakan metrik mean squared error.

- Accuracy_score adalah sebuah metric evaluasi dengan membandingkan jumlah prediksi yang benar dengan jumlah keseluruhan data.
  Jika kita melihat pada tabel confusion metric yang ada di bawah,
dapat kita lihat ada 2 label, yaitu true label (label yang sesungguhnya) dan predicted label (label yang diprediksi oleh model). Accuracy_score akan membandingkan jumlah label 1 yang diprediksi 1 ditambah jumlah label 0 yang diprediksi 0 di bagi jumlah keseluruhan data.


Dimana TP (True Positif) adalah label 1 yang diprediksi 1, dan TN (True Negatif) adalah label 0 yang diprediksi 0.

- mean_squared_error adalah sebuah metric yang digunakan untuk mengukur seberapa besar error / kesalahan dalam prediksi. Semakin kecil error maka performa dari model semakin baik. mean_squared_error akan menghitung selisih rata-rata nilai sebenarnya dengan nilai prediksi.

Dengan metric tersebut didapatkan akurasi sebesar 0,72 dan loss sebesar 0,27.
"""

# standarisasi fitur numerik

X_test.loc[:, numerical_features] = scaler.transform(X_test[numerical_features])

X_test

acc_val = accuracy_score(clf.predict(X_test), y_test)
print('accuracy in validation : ', acc_val)

error_val = mean_squared_error(y_true=y_test, y_pred=clf.predict(X_test))
print('error in validation : ', error_val)

"""Dalam pelatihan model di atas kita mendapatkan akurasi 0,82 pada data train dan 0,74 pada data test

sedangkan errornya kita mendapatkan 0,17 pada data train dan 0,27 pada data test

# confusion metrics
"""

from sklearn.metrics import plot_confusion_matrix

plot_confusion_matrix(clf, X_test, y_test, values_format='d')

"""# prediksi"""

predict_data = X_test.iloc[:20].copy()
pred_dict = {'y_true':y_test[:20]}
pred_dict['prediksi_RandomForestClassifier'] = clf.predict(predict_data)
 
pd.DataFrame(pred_dict)

acc_pred = accuracy_score(clf.predict(predict_data), y_test[:20])
print('accuracy in prediction : ', acc_pred)

error_pred = mean_squared_error(y_true=y_test[:20], y_pred=clf.predict(predict_data))
print('error in prediction : ', error_pred)

"""## Penutup

Demikian adalah rangkuman dari tahapan Prediksi penyakit kardiovaskular yang saya lakukan. Prediksi masih bisa diperbaiki dengan melakukan hypertuning parameter ataupun memperbaiki dengan memeperbaiki kualitas data.
"""