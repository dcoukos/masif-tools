import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Sequential, ReLU
from torch_geometric.nn import GCNConv, FeaStConv, MessagePassing, knn_graph, BatchNorm
import params as p
# graclus, avg_pool_x


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

    def forward(self, x, edge_index):  # Had to modify to work with Tensorboard
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

        return x


class OneConv(torch.nn.Module):
    def __init__(self, n_features, dropout=True):
        # REMEMBER TO UPDATE MODEL NAME
        super(OneConv, self).__init__()
        self.conv1 = FeaStConv(n_features, 16)
        self.lin1 = Linear(16, 8)
        self.out = Linear(8, 1)

    def forward(self, data):
        in_, edge_index = data.x, data.edge_index
        x = self.conv1(in_, edge_index)
        x = x.relu()
        x = self.lin1(x)
        x = x.relu()
        x = self.out(x)
        x = torch.sigmoid(x)

        return x


class TwoConv(torch.nn.Module):
    def __init__(self, n_features, dropout=True):
        # REMEMBER TO UPDATE MODEL NAME
        super(TwoConv, self).__init__()
        self.conv1 = FeaStConv(n_features, 16)
        self.conv2 = FeaStConv(16, 32)
        self.lin1 = Linear(32, 16)
        self.lin2 = Linear(16, 8)
        self.lin3 = Linear(8, 4)
        self.out = Linear(4, 1)

    def forward(self, data):
        in_, edge_index = data.x, data.edge_index
        x = self.conv1(in_, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)
        x = x.relu()
        x = self.out(x)
        x = torch.sigmoid(x)

        return x


class ThreeConv(torch.nn.Module):
    # Seems underpowered, but less epoch-to-epoch variance in prediction compared to BasicNet
    # Quick Setup: back to back with max pool and pass through?

    # TODO: confirm that linear layers defined below are functionally equivalent to 1x1 conv

    def __init__(self, n_features, heads=1, dropout=True):
        super(ThreeConv, self).__init__()
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        self.conv2 = FeaStConv(16, 32, heads=heads)
        self.conv3 = FeaStConv(32, 64, heads=heads)
        self.lin1 = Linear(64, 32)
        self.lin2 = Linear(32, 16)
        self.lin3 = Linear(16, 8)
        self.lin4 = Linear(8, 4)
        self.out = Linear(4, 1)
        self.drop_bool = dropout
        self.dropout = Dropout(p=0.3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        in_, edge_index = data.x, data.edge_index
        x = self.conv1(in_, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.lin1(x)
        x = self.dropout(x) if self.drop_bool else x
        x = x.relu()
        x = self.lin2(x)
        x = self.dropout(x) if self.drop_bool else x
        x = x.relu()
        x = self.lin3(x)
        x = self.dropout(x) if self.drop_bool else x
        x = x.relu()
        x = self.lin4(x)
        x = self.dropout(x) if self.drop_bool else x
        x = x.relu()
        x = self.out(x)
        x = torch.sigmoid(x)

        return x


class SixConv(torch.nn.Module):
    # Seems underpowered, but less epoch-to-epoch variance in prediction compared to BasicNet
    # Quick Setup: back to back with max pool and pass through?

    # TODO: confirm that linear layers defined below are functionally equivalent to 1x1 conv

    def __init__(self, n_features, heads=1, dropout=True):
        super(SixConv, self).__init__()
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv2 = FeaStConv(16, 16, heads=heads)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.conv3 = FeaStConv(16, 16, heads=heads)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        self.conv4 = FeaStConv(16, 16)
        torch.nn.init.xavier_uniform(self.conv4.weight)
        self.conv5 = FeaStConv(16, 32)
        torch.nn.init.xavier_uniform(self.conv5.weight)
        self.conv6 = FeaStConv(32, 64)
        torch.nn.init.xavier_uniform(self.conv6.weight)
        self.lin1 = Linear(64, 16)
        self.lin2 = Linear(16, 4)
        self.out = Linear(4, 1)
        self.drop_bool = dropout
        self.dropout = Dropout(p=0.3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = self.conv5(x, edge_index)
        x = x.relu()
        x = self.conv6(x, edge_index)
        x = x.relu()
        x = self.lin1(x)
        x = self.dropout(x) if self.drop_bool else x
        x = x.relu()
        x = self.lin2(x)
        x = self.dropout(x) if self.drop_bool else x
        x = x.relu()
        x = self.out(x)
        x = torch.sigmoid(x)

        return x


class SixConvPassThrough(torch.nn.Module):
    # Seems underpowered, but less epoch-to-epoch variance in prediction compared to BasicNet
    # Quick Setup: back to back with max pool and pass through?

    # TODO: confirm that linear layers defined below are functionally equivalent to 1x1 conv

    def __init__(self, n_features, heads=1, dropout=True):
        super(SixConvPassThrough, self).__init__()
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv2 = FeaStConv(16, 16, heads=heads)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.conv3 = FeaStConv(16, 16, heads=heads)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        self.conv4 = FeaStConv(16, 16)
        torch.nn.init.xavier_uniform(self.conv4.weight)
        self.conv5 = FeaStConv(16, 32)
        torch.nn.init.xavier_uniform(self.conv5.weight)
        self.conv6 = FeaStConv(32, 64)
        torch.nn.init.xavier_uniform(self.conv6.weight)
        self.lin1 = Linear(96, 16)
        self.lin2 = Linear(16, 4)
        self.out = Linear(4, 1)
        self.drop_bool = dropout
        self.dropout = Dropout(p=0.3)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        in_, edge_index = data.x, data.edge_index
        x1 = self.conv1(in_, edge_index)
        x1 = x1.relu()
        x2 = self.conv2(x1, edge_index)
        x2 = x2.relu()
        x2 = self.conv3(x2, edge_index)
        x2 = x2.relu()
        x3 = self.conv4(x2, edge_index)
        x3 = x3.relu()
        x3 = self.conv5(x3, edge_index)
        x3 = x3.relu()
        x3 = self.conv6(x3, edge_index)
        x4 = torch.cat((x1, x2, x3), dim=1)
        x4 = x4.relu()
        x4 = self.lin1(x4)
        x5 = self.dropout(x4) if self.drop_bool else x4
        x5 = x5.relu()
        x5 = self.lin2(x5)
        x6 = self.dropout(x5) if self.drop_bool else x5
        x6 = x6.relu()
        x6 = self.out(x6)
        x6 = torch.sigmoid(x6)

        return x6


class SixConvPT_LFC(torch.nn.Module):
    # Seems underpowered, but less epoch-to-epoch variance in prediction compared to BasicNet
    # Quick Setup: back to back with max pool and pass through?

    # TODO: confirm that linear layers defined below are functionally equivalent to 1x1 conv

    def __init__(self, n_features, heads=1, dropout=True):
        super(SixConvPT_LFC, self).__init__()
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv2 = FeaStConv(16, 16, heads=heads)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.conv3 = FeaStConv(16, 16, heads=heads)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        self.conv4 = FeaStConv(16, 16)
        torch.nn.init.xavier_uniform(self.conv4.weight)
        self.conv5 = FeaStConv(16, 32)
        torch.nn.init.xavier_uniform(self.conv5.weight)
        self.conv6 = FeaStConv(32, 64)
        torch.nn.init.xavier_uniform(self.conv6.weight)
        self.lin1 = Linear(96, 256)
        self.lin2 = Linear(256, 64)
        self.lin3 = Linear(64, 16)
        self.out = Linear(16, 1)
        self.drop_bool = dropout
        self.dropout = Dropout(p=0.5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        in_, edge_index = data.x, data.edge_index
        x1 = self.conv1(in_, edge_index)
        x1 = x1.relu()
        x2 = self.conv2(x1, edge_index)
        x2 = x2.relu()
        x2 = self.conv3(x2, edge_index)
        x2 = x2.relu()
        x3 = self.conv4(x2, edge_index)
        x3 = x3.relu()
        x3 = self.conv5(x3, edge_index)
        x3 = x3.relu()
        x3 = self.conv6(x3, edge_index)
        x4 = torch.cat((x1, x2, x3), dim=1)
        x4 = x4.relu()
        x4 = self.lin1(x4)
        x4 = self.dropout(x4) if self.drop_bool else x4
        x4 = x4.relu()
        x4 = self.lin2(x4)
        x4 = self.dropout(x4) if self.drop_bool else x4
        x4 = x4.relu()
        x4 = self.lin3(x4)
        x4 = self.dropout(x4) if self.drop_bool else x4
        x4 = x4.relu()
        x4 = self.out(x4)
        x4 = torch.sigmoid(x4)

        return x4


class SixConvResidual(torch.nn.Module):
    # Seems underpowered, but less epoch-to-epoch variance in prediction compared to BasicNet
    # Quick Setup: back to back with max pool and pass through?

    # TODO: confirm that linear layers defined below are functionally equivalent to 1x1 conv

    def __init__(self, n_features, heads=1, dropout=True):
        super(SixConvResidual, self).__init__()
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv2 = FeaStConv(16+n_features, 32, heads=heads)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.conv3 = FeaStConv(48 + n_features, 32, heads=heads)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        self.conv4 = FeaStConv(80 + n_features, 32)
        torch.nn.init.xavier_uniform(self.conv4.weight)
        self.conv5 = FeaStConv(112 + n_features, 32)
        torch.nn.init.xavier_uniform(self.conv5.weight)
        self.conv6 = FeaStConv(144 + n_features, 64)
        torch.nn.init.xavier_uniform(self.conv6.weight)
        self.lin1 = Linear(208 + n_features, 256)
        self.lin2 = Linear(256, 64)
        self.lin3 = Linear(64, 16)
        self.out = Linear(16, 1)
        self.batch1 = BatchNorm(16+n_features)
        self.batch2 = BatchNorm(80+n_features)
        self.batch3 = BatchNorm(208+n_features)
        self.dropout = Dropout(p=0.5)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        in_, edge_index = data.x, data.edge_index
        x1 = self.conv1(in_, edge_index)
        x1 = x1.relu()
        cat0 = torch.cat((x1, in_), dim=1)
        x2 = self.conv2(cat0, edge_index)
        x2 = self.batch1(x2) if p.batchnorm else x2
        x2 = x2.relu()
        cat1 = torch.cat((cat0, x2), dim=1)
        x3 = self.conv3(cat1, edge_index)
        x3 = x3.relu()
        cat2 = torch.cat((cat1, x3), dim=1)
        x4 = self.conv4(cat2, edge_index)
        x4 = self.batch2(x4) if p.batchnorm else x4
        x4 = x4.relu()
        cat3 = torch.cat((cat2, x4), dim=1)
        x5 = self.conv5(cat3, edge_index)
        x5 = x5.relu()
        cat4 = torch.cat((cat3, x5), dim=1)
        x6 = self.conv6(cat4, edge_index)
        z = torch.cat((cat4, x6), dim=1)
        z = self.batch3(z) if p.batchnorm else z
        z = z.relu()
        z = self.lin1(z)
        z = self.dropout(z) if p.dropout else z
        z = z.relu()
        z = self.lin2(z)
        z = self.dropout(z) if p.dropout else z
        z = z.relu()
        z = self.lin3(z)
        z = self.dropout(z) if p.dropout else z
        z = z.relu()
        z = self.out(z)
        z = torch.sigmoid(z)

        return z


class ThreeConvGlobal(torch.nn.Module):
    '''
    '''

    def __init__(self, n_features, heads=1, dropout=True):
        super(ThreeConvGlobal, self).__init__()
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        self.conv2 = FeaStConv(16, 32, heads=heads)
        self.conv3 = FeaStConv(64, 64, heads=heads)
        self.edge1 = DynamicEdgeConv(16, 32)
        self.edge2 = DynamicEdgeConv(16, 32)
        self.lin1 = Linear(96, 96)
        self.lin2 = Linear(96, 32)
        self.lin3 = Linear(32, 8)
        self.out = Linear(8, 1)
        self.drop_bool = dropout
        self.dropout = Dropout(p=0.4)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        in_, edge_index = data.x, data.edge_index
        x = self.conv1(in_, edge_index)
        x = x.relu()
        y = self.edge1(x)
        y = y.relu()
        cat0 = torch.cat((x, y), dim=1)
        x = self.conv2(x, edge_index)
        x = x.relu()
        y = self.edge2(y)
        y = y.relu()
        x = self.conv3(cat0, edge_index)
        x = x.relu()
        cat1 = torch.cat((x, y), dim=1)
        x = self.lin1(cat1)
        x = self.dropout(x) if self.drop_bool else x
        x = x.relu()
        x = self.lin2(x)
        x = self.dropout(x) if self.drop_bool else x
        x = x.relu()
        x = self.lin3(x)
        x = self.dropout(x) if self.drop_bool else x
        x = x.relu()
        x = x.relu()
        x = self.out(x)
        x = torch.sigmoid(x)

        return x


class EdgeConv(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(EdgeConv, self).__init__(aggr='max') #  "Max" aggregation.
        self.mlp = Sequential(Linear(2 * in_channels, out_channels),
                              ReLU(),
                              Linear(out_channels, out_channels))

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]

        return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

    def message(self, x_i, x_j):
        # x_i has shape [E, in_channels]
        # x_j has shape [E, in_channels]

        tmp = torch.cat([x_i, x_j - x_i], dim=1)  # tmp has shape [E, 2 * in_channels]
        return self.mlp(tmp)

    def update(self, aggr_out):
        # aggr_out has shape [N, out_channels]

        return aggr_out


class DynamicEdgeConv(EdgeConv):
    def __init__(self, in_channels, out_channels, k=6):
        super(DynamicEdgeConv, self).__init__(in_channels, out_channels)
        self.k = k

    def forward(self, x, batch=None):
        edge_index = knn_graph(x, self.k, batch, loop=False, flow=self.flow)
        return super(DynamicEdgeConv, self).forward(x, edge_index)
