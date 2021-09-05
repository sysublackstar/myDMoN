import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import torch
from networkx.readwrite.gml import read_gml
from torch.optim import Adam, SGD, RMSprop
from torch_geometric.data import Data
from tqdm import tqdm
import seaborn as sns
import metrics
from model import MyModel
from utils import convert_scipy_sparse_to_sparse_tensor
from torchviz import make_dot
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
    # num_channels = 8
    num_channels = 64
    collapse_regularization = 1
    plot_epoch_modularity = True
    # graph_mode = 'knn'

    graph_mode = 'knn'
    if graph_mode == 'epsilon':
        adjacency_raw, features = load_data('data/graph.gml', 'data/features.npy')
    elif graph_mode == 'knn':
        adjacency_raw, features = load_data('data/k_10.gml', 'data/features.npy')

    graph = gen_graph(adjacency_raw, features)
    adjacency = convert_scipy_sparse_to_sparse_tensor(adjacency_raw)

    features, edges = graph.x, graph.edge_index
    for num_clusters in [8]:
        df = pd.read_csv('data/data.csv')
        model = MyModel('gat', num_channels, num_clusters, collapse_regularization=collapse_regularization)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model.to(device)

        if device == torch.device('cuda'):
            features = features.cuda()
            edges = edges.cuda()
            adjacency = adjacency.cuda()

        optimizer = Adam(model.parameters(), lr=0.001)
        # optimizer = SGD(model.parameters(), lr=0.001, momentum=0.9)
        # optimizer = RAdam(model.parameters(), lr=0.001)
        model.train()
        # plot result
        for epoch in tqdm(range(epochs)):
            optimizer.zero_grad()
            _ = model(features, edges, adjacency)
            loss = model.dmon.loss
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
            # print(f'epoch {epoch + 1}, losses: ' + str(loss.item()))
                features_pooled, assignments = model(features, edges, adjacency)
                assignments = assignments.cpu().detach().numpy()
                cur_clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters.
            # Prints some metrics used in the paper.
            # print('Conductance:', metrics.conductance(adjacency_raw, cur_clusters))
            # print('Modularity:', metrics.modularity(adjacency_raw, cur_clusters))
            # modularity = metrics.modularity(adjacency_raw, cur_clusters)
            # if modularity > max_modularity:
            #     print(modularity)
            #     max_modularity = modularity
            #     clusters = cur_clusters
                plt.figure()
                plt.scatter(df['X'], df['Y'], c=cur_clusters, s=1)
                plt.xticks([])
                plt.yticks([])
                plt.savefig(f"fig/gat_{num_clusters}_clusters.jpg", dpi=300)
                plt.show()
        df['clusters'] = cur_clusters
        df.to_csv(f"{num_clusters}.csv", index=False)
    # plot modularity
    # model = MyModel('gat', num_channels, num_clusters)
    # mod = []
    # ep = list(range(1, epochs + 1))
    # device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # # device = torch.device('cpu')
    # model.to(device)
    # optimizer = Adam(model.parameters(), lr=0.001)
    # model.train()
    # for epoch in tqdm(range(epochs)):
    #     optimizer.zero_grad()
    #     _ = model(features, edges, adjacency)
    #     loss = model.dmon.loss
    #     loss.backward()
    #     optimizer.step()
    #     _, assignments = model(features, edges, adjacency)
    #     assignments = assignments.cpu().detach().numpy()
    #     clusters = assignments.argmax(axis=1)  # Convert soft to hard clusters.
    #     mod.append(metrics.modularity(adjacency_raw, clusters))
    # func = interp1d(ep, mod, kind='cubic')
    # ep_new = np.arange(1, 100, 0.1)
    # mod_new = func(ep_new)
    # plt.xlabel('epochs')
    # plt.ylabel('modularity')
    # plt.plot(ep_new, mod_new)
    # plt.xticks(np.linspace(0, 100, 11))
    # plt.xlim((0, 100))
    # plt.show()

    # plot covariance
    # model.output.detach.numpy()
    # 需要随机采样一些点 不然不好看
    # arr = np.hstack([npep.array(clusters).reshape(7237, 1), model.output.cpu().detach().numpy()])
    # arr = arr[np.argsort(arr[:, 0])]
    # sample = []
    # for i in range(num_clusters):
    #     sample_i = np.random.choice(np.where(arr[:, 0] == i)[0], 100, replace=True)
    #     sample.extend(sample_i.tolist())
    # arr = arr[sample, :]
    # # arr = arr[:, 1:]
    # corr = np.corrcoef(arr)
    # # sns.heatmap(corr, xticklabels=False, yticklabels=False, cbar=False)
    # sns.heatmap(corr)
    # plt.show()

    # plot covariance
    # model.output.detach.numpy()
    # 需要随机采样一些点 不然不好看
    # arr = np.hstack([np.array(clusters).reshape(7237, 1), features_pooled.cpu().detach().numpy()])
    # arr = arr[np.argsort(arr[:, 0])]
    # sample = []
    # for i in range(num_clusters):
    #     sample_i = np.random.choice(np.where(arr[:, 0] == i)[0], 100, replace=True)
    #     sample.extend(sample_i.tolist())
    # arr = arr[sample, :]
    # # arr = arr[:, 1:]
    # corr = np.corrcoef(arr)
    # # sns.heatmap(corr, xticklabels=False, yticklabels=False, cbar=False)
    # sns.heatmap(corr)
    # plt.show()
    #
