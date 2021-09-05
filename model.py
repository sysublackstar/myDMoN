import torch.nn as nn
from gnn import GCN, GAT
from dmon import DMoN


class MyModel(nn.Module):
    def __init__(self,
                 gnn,
                 n_channels,
                 n_clusters,
                 collapse_regularization=1):
        """Initializes the layer with specified parameters."""
        super(MyModel, self).__init__()

        if gnn == 'gcn':
            self.gnn = GCN(n_channels)
        elif gnn == 'gat':
            self.gnn = GAT(n_channels)
        self.dmon = DMoN(n_clusters, n_channels, collapse_regularization)

    def forward(self, features, edges, adjacency):
        output = self.gnn(features, edges)
        pool, pool_assignments = self.dmon(output, adjacency)
        return pool, pool_assignments

