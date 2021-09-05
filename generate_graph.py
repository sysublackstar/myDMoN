import networkx as nx
import pandas as pd
from networkx.readwrite.gml import write_gml
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.preprocessing import MinMaxScaler
from tqdm import tqdm
import numpy as np

df = pd.read_csv("./data/data.csv")
n = df.shape[0]
df_scaled = MinMaxScaler().fit_transform(df.drop(['X', 'Y'], axis=1))

cs = cosine_similarity(df_scaled)

G = nx.Graph()
G.add_nodes_from(list(range(n)))

# epsilon_graph
distance_thresh = 100
cs_thresh = 0.99
# add edge
for i in tqdm(range(n)):
    x = df.loc[i, 'X']
    y = df.loc[i, 'Y']
    for j in range(i):
        xx = df.loc[j, 'X']
        yy = df.loc[j, 'Y']
        # if (x-xx)**2 + (y-yy)**2 < 300 and cs[i][j] > 0.995:

        if (x - xx) ** 2 + (y - yy) ** 2 < distance_thresh and cs[i][j] > cs_thresh:
            G.add_edge(i, j)

write_gml(G, f'./data/graph_{distance_thresh}_{cs_thresh * 100}.gml')


# knn graph features

# k = 20
# for i in tqdm(range(n)):
#     x = df.loc[i, 'X']
#     y = df.loc[i, 'Y']
#     arr = cs[i]
#     idxs = arr.argsort()[::-1][1: k+1]
#     for j in idxs:
#         xx = df.loc[j, 'X']
#         yy = df.loc[j, 'Y']
#         if (x - xx) ** 2 + (y - yy) ** 2 < distance_thresh:
#             G.add_edge(i, j)
# write_gml(G, f'./data/graph_k={k}.gml')

# knn distance
distance = euclidean_distances(df[['X', 'Y']])
for k in [4, 6, 10, 16]:
    for i in tqdm(range(n)):
        idxs = distance[i].argsort()[1: k+1]
        for j in idxs:
            G.add_edge(i, j)
    write_gml(G, f'./data/k_{k}.gml')
