import pandas as pd
import numpy as np
import skfuzzy as fuzz
from sklearn.datasets import make_blobs
import matplotlib.pyplot as plt

def load_dataset(file_path):
    # Load dataset from CSV file
    dataset = pd.read_csv(file_path)
    return dataset.values

def fuzzy_c_means(data, num_clusters, max_iters=100, error=0.005):
    # Fuzzy C-Means clustering
    cntr, u, _, _, _, _, _ = fuzz.cluster.cmeans(
        data.T, num_clusters, 2, error, max_iters)
    return cntr, u

def plot_clusters(data, u, centers):
    # Plot the clusters
    cluster_membership = np.argmax(u, axis=0)
    colors = ['b', 'g', 'r', 'c', 'm', 'y', 'k']

    for i in range(len(centers)):
        plt.scatter(data[cluster_membership == i, 0],
                    data[cluster_membership == i, 1], c=colors[i], s=30, label=f'Cluster {i}')

    plt.scatter(centers[:, 0], centers[:, 1], c='black', marker='x', s=100, label='Centroids')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.title('Fuzzy C-Means Clustering')
    plt.legend()
    plt.show()

def main():
    # Load dataset
    file_path = 'Country_data.csv'  # Ganti dengan path file CSV Anda
    data = load_dataset(file_path)

    # Pilih jumlah cluster
    num_clusters = int(input("Masukkan jumlah cluster: "))

    # Jalankan Fuzzy C-Means
    centers, u = fuzzy_c_means(data, num_clusters)

    # Visualisasi hasil clustering
    plot_clusters(data, u, centers)

if __name__ == "__main__":
    main()
