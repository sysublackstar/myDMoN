import pandas as pd
import numpy as np
from sklearn.cluster import DBSCAN, KMeans, SpectralClustering
from IPython.display import SVG

from sknetwork.clustering import modularity
import matplotlib.pyplot as plt
from networkx.readwrite.gml import read_gml
import networkx as nx
from metrics import conductance
from scipy.sparse import csr_matrix
from sknetwork.hierarchy import LouvainHierarchy, BiLouvainHierarchy
from sknetwork.hierarchy import cut_straight, dasgupta_score, tree_sampling_divergence
from sknetwork.visualization import svg_graph, svg_digraph, svg_bigraph, svg_dendrogram


plt.style.use(['science', 'ieee'])

filepath = './data/data.csv'
feat_file = './data/features.npy'
df = pd.read_csv(filepath)
features = np.load(feat_file)

G = read_gml('data/graph_100_99.0.gml')
adj_matrix = csr_matrix(nx.to_numpy_array(G))
louvain_hierarchy = LouvainHierarchy()

hierarchy = louvain_hierarchy.fit_transform(adj_matrix)
image = svg_dendrogram(hierarchy)

n_clusters = 2
clusters = cut_straight(hierarchy, n_clusters=n_clusters)

plt.scatter(df['X'], df['Y'], c=clusters, s=2)
plt.title(f"hierarchy {n_clusters} clusters")
plt.show()
# plt.savefig(f"Spectral Clustering {n_clusters} clusters.jpg", dpi=300)
print(modularity(adj_matrix, clusters))
print(conductance(adj_matrix, clusters))

SVG(image)
plt.show()