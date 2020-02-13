import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Sequential, ReLU
from torch_geometric.nn import GCNConv, FeaStConv, MessagePassing, knn_graph, BatchNorm, TopKPooling
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
    def __init__(self, n_features, heads=4):
        # REMEMBER TO UPDATE MODEL NAME
        super(TwoConv, self).__init__()
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        self.conv2 = FeaStConv(16, 32, heads=heads)
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

    def __init__(self, n_features, heads=4, dropout=True):
        super(ThreeConv, self).__init__()
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        self.conv2 = FeaStConv(16, 32, heads=heads)
        self.conv3 = FeaStConv(32, 64, heads=heads)
        self.batch = BatchNorm(64)
        self.lin1 = Linear(64, 32)
        self.lin2 = Linear(32, 16)
        self.lin3 = Linear(16, 8)
        self.lin4 = Linear(8, 4)
        self.out = Linear(4, 1)

    def forward(self, data):
        in_, edge_index = data.x, data.edge_index
        x = self.conv1(in_, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.batch(x)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)
        x = x.relu()
        x = self.lin4(x)
        x = x.relu()
        x = self.out(x)
        x = torch.sigmoid(x)

        return x


class SixConv(torch.nn.Module):
    # Seems underpowered, but less epoch-to-epoch variance in prediction compared to BasicNet
    # Quick Setup: back to back with max pool and pass through?

    # TODO: confirm that linear layers defined below are functionally equivalent to 1x1 conv

    def __init__(self, n_features, heads=4, dropout=True):
        super(SixConv, self).__init__()
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        torch.nn.init.xavier_uniform(self.conv1.weight)
        self.conv2 = FeaStConv(16, 16, heads=heads)
        torch.nn.init.xavier_uniform(self.conv2.weight)
        self.conv3 = FeaStConv(16, 16, heads=heads)
        torch.nn.init.xavier_uniform(self.conv3.weight)
        self.conv4 = FeaStConv(16, 16)
        torch.nn.init.xavier_uniform(self.conv4.weight)
        self.conv5 = FeaStConv(16, 16)
        torch.nn.init.xavier_uniform(self.conv5.weight)
        self.conv6 = FeaStConv(16, 16)
        torch.nn.init.xavier_uniform(self.conv6.weight)
        self.lin1 = Linear(16, 64)
        self.lin2 = Linear(64, 64)
        self.lin3 = Linear(64, 16)
        self.out = Linear(16, 1)
        self.batch1 = BatchNorm(16)
        self.batch2 = BatchNorm(16)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x2 = x
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.batch1(x)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()
        x4 = x
        x = self.conv5(x, edge_index)
        x = x.relu()
        x = self.conv6(x, edge_index)
        x = self.batch2(x)
        x = x.relu()
        x = x + x2 + x4
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = self.lin3(x)
        x = x.relu()
        x = self.out(x)
        x = torch.sigmoid(x)

        return x


class SixConvPassThrough(torch.nn.Module):
    # Seems underpowered, but less epoch-to-epoch variance in prediction compared to BasicNet
    # Quick Setup: back to back with max pool and pass through?

    # TODO: confirm that linear layers defined below are functionally equivalent to 1x1 conv

    def __init__(self, n_features, heads=4, dropout=True):
        super(SixConvPassThrough, self).__init__()
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        self.conv2 = FeaStConv(16, 32, heads=heads)
        self.conv3 = FeaStConv(32, 32, heads=heads)
        self.conv4 = FeaStConv(32, 32, heads=heads)
        self.conv5 = FeaStConv(32, 32, heads=heads)
        self.conv6 = FeaStConv(32, 64, heads=heads)
        self.lin1 = Linear(128, 256)
        self.lin2 = Linear(256, 64)
        self.lin3 = Linear(64, 16)
        self.out = Linear(16, 1)
        self.batch1 = BatchNorm(32)
        self.batch2 = BatchNorm(32)
        self.batch3 = BatchNorm(64)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def forward(self, data):
        in_, edge_index = data.x, data.edge_index
        x1 = self.conv1(in_, edge_index)
        x1 = x1.relu()
        x1 = self.conv2(x1, edge_index)
        x1 = x1.relu()
        x1 = self.batch1(x1)
        x2 = self.conv3(x2, edge_index)
        x2 = x2.relu()
        x2 = self.conv4(x2, edge_index)
        x2 = x2.relu()
        x2 = self.batch2(x2)
        x3 = self.conv5(x2, edge_index)
        x3 = x3.relu()
        x3 = self.conv6(x3, edge_index)
        x3 = x3.relu()
        x3 = self.batch3(x3)
        z = torch.cat((x1, x2, x3), dim=1)
        z = z.relu()
        z = self.lin1(z)
        z = z.relu()
        z = self.lin2(z)
        z = z.relu()
        z = self.lin3(z)
        z = z.relu()
        z = self.out(z)
        z = torch.sigmoid(z)

        return z


class SixConvResidual(torch.nn.Module):
    def __init__(self, n_features, heads=4):
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
        self.batch1 = BatchNorm(32)
        self.batch2 = BatchNorm(32)
        self.batch3 = BatchNorm(208+n_features)

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


class TwentyConv(torch.nn.Module):

    def __init__(self, n_features, heads=4):
        super(TwentyConv, self).__init__()
        self.block1 = FourConvBlock(n_features, 4, heads=heads)
        self.block2 = FourConvBlock(4, 4, heads=heads)
        self.block3 = FourConvBlock(4, 4, heads=heads)
        self.block4 = FourConvBlock(4, 4, heads=heads)
        self.block5 = FourConvBlock(4, 4, heads=heads)
        self.lin1 = Linear(4, 64)
        self.lin2 = Linear(64, 64)
        self.lin3 = Linear(64, 16)
        self.out = Linear(16, 1)

    def forward(self, data):
        # Should edge_index get updated?
        x1, edge_index = data.x, data.edge_index
        x1 = self.block1(x1, edge_index)
        x2 = self.block2(x1, edge_index)
        x2 = x1 + x2
        x3 = self.block3(x2, edge_index)
        x3 = x2 + x3
        x4 = self.block4(x3, edge_index)
        x4 = x3 + x4
        x5 = self.block5(x4, edge_index)
        x5 = x4 + x5
        z = self.lin1(x5)
        z = z.relu()
        z = self.lin2(z)
        z = z.relu()
        z = self.lin3(z)
        z = z.relu()
        z = self.out(z)
        z = torch.sigmoid(z)

        return z


class FourConvBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, heads=4):
        super(FourConvBlock, self).__init__()
        self.conv1 = FeaStConv(in_features, 4, heads=heads)
        self.conv2 = FeaStConv(4, 4, heads=heads)
        self.conv3 = FeaStConv(4, 4, heads=heads)
        self.conv4 = FeaStConv(4, out_features, heads=heads)
        self.batch = BatchNorm(out_features)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = self.batch(x)

        return x


class TwentyPoolConv(torch.nn.Module):

    def __init__(self, n_features, heads=4):
        super(TwentyPoolConv, self).__init__()
        self.block1 = FourConvPoolBlock(n_features, 16, heads=heads)
        self.block2 = FourConvPoolBlock(16, 16, heads=heads)
        self.block3 = FourConvPoolBlock(16, 16, heads=heads)
        self.block4 = FourConvPoolBlock(16, 16, heads=heads)
        self.block5 = FourConvPoolBlock(16, 16, heads=heads)
        self.lin1 = Linear(16, 64)
        self.lin2 = Linear(64, 64)
        self.lin3 = Linear(64, 16)
        self.out = Linear(16, 1)

    def forward(self, data):
        # Should edge_index get updated?
        x1, edge_index = data.x, data.edge_index
        x1 = self.block1(x1, edge_index)
        x2 = self.block2(x1, edge_index)
        x2 = x1 + x2
        x3 = self.block3(x2, edge_index)
        x3 = x2 + x3
        x4 = self.block4(x3, edge_index)
        x4 = x3 + x4
        x5 = self.block5(x4, edge_index)
        x5 = x4 + x5
        z = self.lin1(x5)
        z = z.relu()
        z = self.lin2(z)
        z = z.relu()
        z = self.lin3(z)
        z = z.relu()
        z = self.out(z)
        z = torch.sigmoid(z)

        return z


class FourConvPoolBlock(torch.nn.Module):
    def __init__(self, in_features, out_features, heads=4):
        super(FourConvPoolBlock, self).__init__()
        self.conv1 = FeaStConv(in_features, 16, heads=heads)
        self.conv2 = FeaStConv(16, 16, heads=heads)
        self.conv3 = FeaStConv(16, 16, heads=heads)
        self.conv4 = FeaStConv(16, out_features, heads=heads)
        self.batch = BatchNorm(out_features)
        self.pool = TopKPooling(16)

    def forward(self, x, edge_index):
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = self.pool(x, edge_index)
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = self.batch(x)
