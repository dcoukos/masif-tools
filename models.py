import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.nn import FeaStConv

'''
This file implements the models.

'''


class Basic_Net(torch.nn.Module):
    def __init__(self, n_features):
        super(Basic_Net, self).__init__()
        self.conv1 = GCNConv(n_features, 64)
        self.conv2 = GCNConv(64, 256)
        self.conv3 = GCNConv(256, 1)   # How many outputs?

    def forward(self, data):
        x, edge_index = data.x, data.edge_index

        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training)
        x = self.conv3(x, edge_index)
        x = torch.sigmoid(x)

        return F.binary_cross_entropy(x, target=data.y)


'''

class FeaStNet(torch.nn.Module):
    def __init__(self, n_features)
'''
