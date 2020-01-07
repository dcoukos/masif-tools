import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, FeaStConv, graclus, max_pool, knn_interpolate, TopKPooling
from utils import generate_weights
from torch_sparse import spspmm
from torch_geometric.utils import add_self_loops, sort_edge_index, remove_self_loops
from torch_geometric.utils.repeat import repeat


'''
This file implements the models.


With model weights of 0.2, and 0.8, the model does not appear to train. Weight gradients quickly
peak around 0, and predictions are stabilized at 0.
With model weights of 0.1 and 0.9, the model does not appear to train. Weight gradients are more
widely distributed, but predictions are quickly stabilized to 1.
'''


class BasicNet(torch.nn.Module):
    '''
        Baseline network for interface classification. 3 convolutional layers followed by 2
        linear layers and binary cross entropy for classification.
    '''
    def __init__(self, n_features, dropout=True):
        super(BasicNet, self).__init__()
        self.conv1 = GCNConv(n_features, 64)
        self.conv2 = GCNConv(64, 64)
        self.conv3 = GCNConv(64, 64)
        self.lin1 = Linear(64, 8)
        self.lin2 = Linear(8, 1)
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, edge_index, labels, graph_to_tb=False):  # Had to modify to work with Tensorboard
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) if self.dropout else x
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) if self.dropout else x
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = self.lin1(x)
        x = F.relu(x)
        x = self.lin2(x)
        x = torch.sigmoid(x)

        if type(graph_to_tb) == torch.Tensor:
            loss = F.binary_cross_entropy(x, target=labels)
        else:
            loss = F.binary_cross_entropy(x, target=labels, weight=generate_weights(labels))

        return loss, x


class FeaStNet(torch.nn.Module):
    '''
        Single-scale graph convolutional network based on Verma et al.

        0 - input(3) - LIN(16) - CONV(32) - CONV(64) - CONV(128) - LIN(1024) - Output(50)
    '''
    # TODO: confirm that linear layers defined below are functionally equivalent to 1x1 conv

    def __init__(self, n_features, dropout=True):
        super(FeaStNet, self).__init__()
        self.lin1 = Linear(n_features, 16)
        self.conv1 = FeaStConv(16, 32)
        self.conv2 = FeaStConv(32, 64)
        self.conv3 = FeaStConv(64, 128)
        self.lin2 = Linear(128, 1024)
        self.lin3 = Linear(1024, 1)
        self.dropout = dropout
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, x, edge_index, labels, graph_to_tb=False):
        # Should edge_index be redefined during run?
        x = self.lin1(x)
        x = F.relu(x)
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) if self.dropout else x
        x = self.conv2(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) if self.dropout else x
        x = self.conv3(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) if self.dropout else x
        x = self.lin2(x)
        x = F.relu(x)
        x = F.dropout(x, training=self.training) if self.dropout else x
        x = self.lin3(x)
        x = torch.sigmoid(x)

        if type(graph_to_tb) == torch.Tensor:
            loss = F.binary_cross_entropy(x, target=labels)
        else:
            loss = F.binary_cross_entropy(x, target=labels, weight=generate_weights(labels))

        return loss, x


def custom_pool():
    pass
