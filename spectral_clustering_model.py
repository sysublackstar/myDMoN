import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from sknetwork.clustering import modularity
import matplotlib.pyplot as plt

from metrics import conductance
from scipy.sparse import csr_matrix
from networkx.readwrite.gml import read_gml
import networkx as nx
plt.style.use(['science', 'ieee'])

filepath = './data/data.csv'
feat_file = './data/features.npy'
df = pd.read_csv(filepath)
features = np.load(feat_file)

G = read_gml('data/graph_100_99.0.gml')
adj_matrix = csr_matrix(nx.to_numpy_array(G))

# for n_clusters in [2, 3, 4, 6, 8]:
for n_clusters in [3]:

    '''Spectral Clustering'''
    clusters = SpectralClustering(affinity='precomputed', assign_labels="discretize", random_state=0,
                                  n_clusters=n_clusters).fit_predict(adj_matrix)
    plt.scatter(df['X'], df['Y'], c=clusters, s=2)
    plt.title(f"Spectral Clustering {n_clusters} clusters")
    plt.show()
    # plt.savefig(f"Spectral Clustering {n_clusters} clusters.jpg", dpi=300)
    print(modularity(adj_matrix, clusters))
    print(conductance(adj_matrix, clusters))