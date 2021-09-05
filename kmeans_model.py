import pandas as pd
import numpy as np
from sklearn.cluster import KMeans, SpectralClustering, Birch
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from networkx.readwrite.gml import read_gml
import networkx as nx

plt.style.use(['science', 'ieee'])

filepath = './data/data.csv'
feat_file = './data/features.npy'
df = pd.read_csv(filepath)
features = np.load(feat_file)

# n_clusters = 4
for n_clusters in [3, 4, 6, 8]:
    # kmeans
    model = KMeans(n_clusters=n_clusters)
    clusters = model.fit_predict(features)
    plt.scatter(df['X'], df['Y'], c=clusters, s=1)
    plt.xticks([])
    plt.yticks([])
    # plt.title(f"Kmeans {n_clusters} clustecrs")
    plt.savefig(f"fig/Kmeans_{n_clusters}_clusters", dpi=300)
    plt.show()

    # birch
    model = Birch(threshold=0.005, n_clusters=n_clusters)
    clusters = model.fit_predict(features)
    plt.scatter(df['X'], df['Y'], c=clusters, s=1)
    plt.xticks([])
    plt.yticks([])
    # plt.title(f"Birch {n_clusters} clusters")
    plt.savefig(f"fig/Birch_{n_clusters}_clusters", dpi=300)
    plt.show()

    filepath = './data/data.csv'
    feat_file = './data/features.npy'
    df = pd.read_csv(filepath)
    # features = np.load(feat_file)

    G = read_gml('data/graph_100_99.0.gml')
    adj_matrix = csr_matrix(nx.to_numpy_array(G))
    # sc
    model = SpectralClustering(
        affinity='precomputed', assign_labels="discretize", random_state=0,
        n_clusters=n_clusters)
    clusters = model.fit_predict(adj_matrix)
    plt.xticks([])
    plt.yticks([])
    plt.scatter(df['X'], df['Y'], c=clusters, s=1)
    # plt.title(f"Spectral Clustering {n_clusters} clusters")
    plt.savefig(f"fig/Spectral_Clustering_{n_clusters}_clusters", dpi=300)
    plt.show()
