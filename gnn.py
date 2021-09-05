from torch_geometric.nn import GCNConv, GATConv, APPNP, BatchNorm
import torch.nn as nn
import torch.nn.functional as F


class GCN(nn.Module):
    def __init__(self, n_channel):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(16, 48)
        self.conv2 = GCNConv(48, 32)
        self.fc = nn.Linear(32, n_channel)

    def forward(self, x, edge_index):
        x = x.float()
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = self.fc(x)
        return x


class GAT(nn.Module):
    def __init__(self, n_channel):
        super(GAT, self).__init__()
        self.conv1 = GATConv(16, 32, heads=3)
        self.conv2 = GATConv(32*3, 32, heads=1)
        self.fc = nn.Linear(32, n_channel)

    def forward(self, x, edge_index):
        x = x.float()
        x = self.conv1(x, edge_index)
        # x = F.relu(x)
        x = F.selu(x)
        x = self.conv2(x, edge_index)
        # x = F.relu(x)
        x = F.selu(x)
        x = self.fc(x)
        return x

