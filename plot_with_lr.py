import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from networkx.readwrite.gml import read_gml
from torch.optim import Adam
from torch_geometric.data import Data
from tqdm import tqdm
import seaborn as sns
import metrics
from model import MyModel
from utils import convert_scipy_sparse_to_sparse_tensor
from scipy.interpolate import interp1d

# from torchviz import make_dot
plt.style.use(['science', 'ieee'])
SEED = 0
torch.manual_seed(SEED)


def load_data(graph, feat_file):
    G = read_gml(graph)
    features = np.load(feat_file)
    adjacency = nx.adjacency_matrix(G)
    return adjacency, features


def gen_graph(adjacency, features):
    coo = adjacency.tocoo()
    edges = np.vstack((coo.row, coo.col))
    edges = torch.LongTensor(edges)
    graph = Data(x=torch.from_numpy(features).double(), edge_index=edges)
    return graph


if __name__ == '__main__':
    learning_rate = 0.001
    epochs = 100
    num_clusters = 4
    # num_channels = 8
    num_channels = 16
    mod_list = []
    clusters_list = []
    adjacency_raw, features = load_data('data/k_10.gml', 'data/features.npy')
    graph = gen_graph(adjacency_raw, features)
    adjacency = convert_scipy_sparse_to_sparse_tensor(adjacency_raw)
    features, edges = graph.x, graph.edge_index

    model = MyModel('gat', num_channels, num_clusters)
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    device = torch.device('cpu')
    model.to(device)

    if device == torch.device('cuda'):
        features = features.cuda()
        edges = edges.cuda()
        adjacency = adjacency.cuda()
    # graph_mode = 'knn'

    for lr in [0.1, 0.01, 0.001, 0.0001]:
        model = MyModel('gat', num_channels, num_clusters)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        model.to(device)

        if device == torch.device('cuda'):
            features = features.cuda()
            edges = edges.cuda()
            adjacency = adjacency.cuda()

        model.train()
        # plot modularity
        mod = []
        ep = np.arange(1, epochs+1)
        ep_new = np.arange(1, epochs, 0.1)
        # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        optimizer = Adam(model.parameters(), lr=lr)
        model.train()
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            _ = model(features, edges, adjacency)
            loss = model.dmon.loss
            loss.backward()
            optimizer.step()
            _, assignments = model(features, edges, adjacency)
            assignments = assignments.detach().numpy()
            clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters.
            mod.append(metrics.modularity(adjacency_raw, clusters))
        func = interp1d(ep, mod, kind='cubic')
        mod_list.append(func(ep_new))
        clusters_list.append(clusters)

    plt.figure()
    for i in range(len(mod_list)):
        plt.plot(ep_new, mod_list[i], linewidth=1)
    plt.xlabel('epochs')
    plt.ylabel('modularity')
    plt.xticks(np.linspace(0, epochs, 11))
    plt.xlim((0, epochs))
    plt.yticks(np.linspace(0, 0.6, 7))
    plt.ylim((0, 0.6))
    plt.legend(['lr=0.1', 'lr=0.01', 'lr=0.001', 'lr=0.0001'], ncol=2)
    plt.savefig('lr.jpg', dpi=300)
    plt.show()

    # plt.figure(figsize=(24, 18))

    df = pd.read_csv('data/data.csv')
    dic = {0: 0.1, 1: 0.01, 2: 0.001, 3: 0.0001}
    for i in range(len(clusters_list)):
        plt.figure()
        plt.scatter(df['X'], df['Y'], c=clusters_list[i], s=1)
        plt.xticks([])
        plt.yticks([])
        plt.savefig(f'lr_result_{dic[i]}.jpg', dpi=200)
