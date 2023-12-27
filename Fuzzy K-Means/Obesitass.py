import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Langkah 1: Membaca data
data = pd.read_csv('Obesity_Classification.csv', sep=';')  # Ganti dengan lokasi dan nama file data Anda

# Langkah 2: Persiapan data
features = ['Age', 'Height', 'Weight', 'BMI', 'Gender_Female', 'Gender_Male']
target = 'Keterangan'  # Mengubah target menjadi 'Keterangan'

# Mengubah kolom 'Gender' menjadi representasi numerik menggunakan One-Hot Encoding
data_encoded = pd.get_dummies(data, columns=['Gender'])

X = data_encoded[features]

# Langkah 3: Mengecek dan menetapkan nama kolom target
if target not in data_encoded.columns:
    raise ValueError("Kolom target tidak ditemukan dalam dataset.")

y = data_encoded[target]

# Memisahkan data menjadi set pelatihan dan set validasi
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Pilih jumlah cluster dengan metode elbow
inertia_values = []
possible_clusters = range(1, 11)  # Anda bisa mengubah rentang ini sesuai kebutuhan
for num_clusters in possible_clusters:
    kmeans = KMeans(n_clusters=num_clusters, random_state=42)
    kmeans.fit(X_train)
    inertia_values.append(kmeans.inertia_)

# Menampilkan grafik elbow untuk membantu memilih jumlah cluster
import matplotlib.pyplot as plt

plt.plot(possible_clusters, inertia_values, marker='o')
plt.title('Elbow Method for Optimal Cluster Selection')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia (Within-cluster sum of squares)')
plt.show()

# Meminta pengguna untuk memilih jumlah cluster
selected_num_clusters = int(input("Pilih jumlah cluster yang optimal: "))

# Membuat dan melatih model K-Means dengan jumlah cluster yang dipilih
kmeans = KMeans(n_clusters=selected_num_clusters, random_state=42)
kmeans.fit(X_train)

# Menggunakan model untuk prediksi cluster data validasi
cluster_labels = kmeans.predict(X_val)

# Menampilkan hasil
print("Label cluster untuk set validasi:", cluster_labels)
