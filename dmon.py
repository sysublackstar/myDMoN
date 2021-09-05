from typing import List
from typing import Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from gnn import GCN
from utils import convert_scipy_sparse_to_sparse_tensor
import metrics

class DMoN(nn.Module):
    def __init__(self,
                 n_clusters,
                 n_channels,
                 collapse_regularization=1,
                 do_unpooling=True):
        """Initializes the layer with specified parameters."""
        super(DMoN, self).__init__()
        self.n_clusters = n_clusters
        self.do_unpooling = do_unpooling
        self.fc = nn.Linear(n_channels, n_clusters)
        self.loss = torch.tensor(0., requires_grad=True)
        self.collapse_regularization = collapse_regularization,
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, features, adjacency):
        # print(features.shape, adjacency.shape)
        # assert isinstance(features, tf.Tensor)
        # assert isinstance(adjacency, tf.SparseTensor)
        # assert len(features.shape) == 2
        # assert len(adjacency.shape) == 2
        # assert features.shape[0] == adjacency.shape[0]

        assignments = self.fc(features)
        assignments = F.softmax(assignments, dim=1)

        cluster_sizes = torch.sum(assignments, dim=0)
        assignments_pooling = assignments / cluster_sizes

        # degrees = torch.sum(adjacency.to_dense(), dim=1)
        degrees = torch.sum(adjacency.to_dense(), dim=0)
        degrees = torch.reshape(degrees, (-1, 1))

        number_of_nodes = adjacency.shape[1]
        number_of_edges = torch.sum(degrees)

        graph_pooled = torch.sparse.mm(adjacency, assignments).t()
        graph_pooled = torch.mm(graph_pooled, assignments)

        normalizer_left = torch.mm(torch.t(assignments), degrees)
        normalizer_right = torch.mm(torch.t(degrees), assignments)

        normalizer = torch.mm(normalizer_left, normalizer_right) / 2 / number_of_edges

        spectral_loss = - torch.trace(graph_pooled - normalizer) / 2 / number_of_edges

        # collapse_loss = torch.norm(cluster_sizes) / number_of_nodes * np.sqrt(self.n_clusters) - 1
        # collapse_loss = torch.norm(cluster_sizes-torch.mean(cluster_sizes), p=1) / number_of_nodes * np.sqrt(self.n_clusters) / (np.sqrt(self.n_clusters)-1) / 2
        # collapse_loss = torch.norm(cluster_sizes, 2) / number_of_nodes * np.sqrt(self.n_clusters)
        collapse_loss = torch.norm(cluster_sizes - number_of_nodes / self.n_clusters, p=1) / number_of_nodes * np.sqrt(self.n_clusters) / (np.sqrt(self.n_clusters)-1) / 2


        # collapse_loss = collapse_loss.unsqueeze(1)
        # self.loss = spectral_loss
        self.loss = torch.add(spectral_loss, self.collapse_regularization[0] * collapse_loss)

        features_pooled = torch.mm(torch.t(assignments_pooling), features)
        features_pooled = F.selu(features_pooled)
        if self.do_unpooling:
            features_pooled = torch.mm(assignments_pooling, features_pooled)
        return features_pooled, assignments
