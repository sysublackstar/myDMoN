from sknetwork.clustering import Louvain, modularity, KMeans
import networkx as nx
from networkx.readwrite.gml import read_gml
import pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix
from sknetwork.embedding import GSVD
from metrics import conductance

plt.style.use(['science', 'ieee'])

G = read_gml('data/graph_100_99.0.gml')
adj_matrix = csr_matrix(nx.to_numpy_array(G))
# louvain = Louvain(n_aggregations=10)
louvain = KMeans(n_clusters=4)

clusters = louvain.fit_transform(adj_matrix)
df = pd.read_csv('data/data.csv')

plt.scatter(df['X'], df['Y'], c=clusters, s=2)
plt.title(f"Louvain")
plt.show()
print(modularity(adj_matrix, clusters))
print(conductance(adj_matrix, clusters))
