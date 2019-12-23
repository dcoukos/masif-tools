import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv
from utils import generate_weights

'''
This file implements the models.


With model weights of 0.2, and 0.8, the model does not appear to train. Weight gradients quickly
peak around 0, and predictions are stabilized at 0.
With model weights of 0.1 and 0.9, the model does not appear to train. Weight gradients are more
widely distributed, but predictions are quickly stabilized to 1.
'''


class Basic_Net(torch.nn.Module):
    def __init__(self, n_features, dropout=True):
        super(Basic_Net, self).__init__()
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


'''

class FeaStNet(torch.nn.Module):
    def __init__(self, n_features)
'''
