import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Sequential, ReLU, SELU
from torch_geometric.nn import (GCNConv, FeaStConv, MessagePassing, knn_graph, BatchNorm,
                                TopKPooling, graclus, max_pool, knn_interpolate, SAGEConv,
                                TAGConv)

from torch_geometric.data import Data
# graclus, avg_pool_x


'''
This file implements the models.


With model weights of 0.2, and 0.8, the model does not appear to train. Weight gradients quickly
peak around 0, and predictions are stabilized at 0.
With model weights of 0.1 and 0.9, the model does not appear to train. Weight gradients are more
widely distributed, but predictions are quickly stabilized to 1.
'''


class Lens(torch.nn.Module):
    def __init__(self, n_features):
        super(Lens, self).__init__()
        self.spec1 = TAGConv(n_features, 16)
        self.spec2 = TAGConv(16, 16)
        self.spec3 = TAGConv(16, 16)
        self.lin1 = Linear(16, 64)
        self.lin2 = Linear(64, 8)
        self.out = Linear(8, 1)
        self.s1 = SELU()
        self.s2 = SELU()
        self.s3 = SELU()
        self.s4 = SELU()
        self.s5 = SELU()

    def forward(self, data):
        x, edge_index = data.x, data.edge_index


class Spectral(torch.nn.Module):
    '''
        Implements a multi-layer spectral graph cnn. Hope is that longer-distance interactions
        are encoded in the lower frequency spectra.
    '''
    def __init__(self, n_features):
        super(Spectral, self).__init__()

    def forward(self, data):


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
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.lin1(x)
        x = x.relu()
        x = self.out(x)
        x = torch.sigmoid(x)

        return x


class TwoConv(torch.nn.Module):
    def __init__(self, n_features, heads=4, masif_descr=False):
        # REMEMBER TO UPDATE MODEL NAME
        super(TwoConv, self).__init__()
        self.masif_descr = masif_descr
        if masif_descr is True:
            self.pre_lin = Linear(80, n_features)
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        self.conv2 = FeaStConv(16, 16, heads=heads)
        self.lin1 = Linear(16, 16)
        self.lin2 = Linear(16, 4)
        self.s1 = SELU()
        self.s2 = SELU()
        self.s3 = SELU()
        self.s4 = SELU()
        self.out = Linear(4, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.pre_lin(x) if self.masif_descr else x
        x = self.conv1(x, edge_index)
        x = self.s1(x)
        x = self.conv2(x, edge_index)
        x = self.s2(x)
        x = self.lin1(x)
        x = self.s3(x)
        x = self.lin2(x)
        x = self.s4(x)
        x = self.out(x)
        x = torch.sigmoid(x)

        return x


class FourConv(torch.nn.Module):
    def __init__(self, n_features, heads=4):
        # REMEMBER TO UPDATE MODEL NAME
        super(FourConv, self).__init__()
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        self.conv2 = FeaStConv(16, 16, heads=heads)
        self.conv3 = FeaStConv(16, 16, heads=heads)
        self.conv4 = FeaStConv(16, 16, heads=heads)
        self.lin1 = Linear(16, 16)
        self.lin2 = Linear(16, 4)
        self.out = Linear(4, 1)
        self.s1 = SELU()
        self.s2 = SELU()
        self.s3 = SELU()
        self.s4 = SELU()
        self.s5 = SELU()
        self.s6 = SELU()

    def forward(self, data, edge_index=None):
        if edge_index is None:
            x, edge_index = data.x, data.edge_index
        else:
            x = data
        x = self.conv1(x, edge_index)
        x = self.s1(x)
        x = self.conv2(x, edge_index)
        x = self.s2(x)
        x = self.conv3(x, edge_index)
        x = self.s3(x)
        x = self.conv4(x, edge_index)
        x = self.s4(x)
        x = self.lin1(x)
        x = self.s5(x)
        x = self.lin2(x)
        x = self.s6(x)
        x = self.out(x)
        x = torch.sigmoid(x)

        return x


class SixConv(torch.nn.Module):
    def __init__(self, n_features, heads=4, masif_descr=False):
        # REMEMBER TO UPDATE MODEL NAME
        super(SixConv, self).__init__()
        self.masif_descr = masif_descr
        if masif_descr is True:
            self.pre_lin = Linear(80, n_features)
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        self.conv2 = FeaStConv(16, 16, heads=heads)
        self.conv3 = FeaStConv(16, 16, heads=heads)
        self.conv4 = FeaStConv(16, 16, heads=heads)
        self.conv5 = FeaStConv(16, 16, heads=heads)
        self.conv6 = FeaStConv(16, 16, heads=heads)
        self.lin1 = Linear(16, 16)
        self.lin2 = Linear(16, 4)
        self.s1 = SELU()
        self.s2 = SELU()
        self.s3 = SELU()
        self.s4 = SELU()
        self.s5 = SELU()
        self.s6 = SELU()
        self.s7 = SELU()
        self.s8 = SELU()
        self.out = Linear(4, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.pre_lin(x) if self.masif_descr else x
        x = self.conv1(x, edge_index)
        x = self.s1(x)
        x = self.conv2(x, edge_index)
        x = self.s2(x)
        x = self.conv3(x, edge_index)
        x = self.s3(x)
        x = self.conv4(x, edge_index)
        x = self.s4(x)
        x = self.conv5(x, edge_index)
        x = self.s5(x)
        x = self.conv6(x, edge_index)
        x = self.s6(x)
        x = self.lin1(x)
        x = self.s7(x)
        x = self.lin2(x)
        x = self.s8(x)
        x = self.out(x)
        x = torch.sigmoid(x)

        return x


class EightConv(torch.nn.Module):
    def __init__(self, n_features, heads=4):
        # REMEMBER TO UPDATE MODEL NAME
        super(EightConv, self).__init__()
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        self.conv2 = FeaStConv(16, 16, heads=heads)
        self.conv3 = FeaStConv(16, 16, heads=heads)
        self.conv4 = FeaStConv(16, 16, heads=heads)
        self.conv5 = FeaStConv(16, 16, heads=heads)
        self.conv6 = FeaStConv(16, 16, heads=heads)
        self.conv7 = FeaStConv(16, 16, heads=heads)
        self.conv8 = FeaStConv(16, 16, heads=heads)
        self.lin1 = Linear(16, 16)
        self.lin2 = Linear(16, 4)
        self.s1 = SELU()
        self.s2 = SELU()
        self.s3 = SELU()
        self.s4 = SELU()
        self.s5 = SELU()
        self.s6 = SELU()
        self.s7 = SELU()
        self.s8 = SELU()
        self.s9 = SELU()
        self.s10 = SELU()
        self.out = Linear(4, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = self.s1(x)
        x = self.conv2(x, edge_index)
        x = self.s2(x)
        x = self.conv3(x, edge_index)
        x = self.s3(x)
        x = self.conv4(x, edge_index)
        x = self.s4(x)
        x = self.conv5(x, edge_index)
        x = self.s5(x)
        x = self.conv6(x, edge_index)
        x = self.s6(x)
        x = self.conv7(x, edge_index)
        x = self.s7(x)
        x = self.conv8(x, edge_index)
        x = self.s8(x)
        x = self.lin1(x)
        x = self.s9(x)
        x = self.lin2(x)
        x = self.s10(x)
        x = self.out(x)
        x = torch.sigmoid(x)

        return x


class TenConv(torch.nn.Module):
    def __init__(self, n_features, heads=4, masif_descr=False):
        # REMEMBER TO UPDATE MODEL NAME
        super(TenConv, self).__init__()
        self.masif_descr = masif_descr
        if masif_descr is True:
            self.pre_lin = Linear(80, n_features)
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        self.conv2 = FeaStConv(16, 16, heads=heads)
        self.conv3 = FeaStConv(16, 16, heads=heads)
        self.conv4 = FeaStConv(16, 16, heads=heads)
        self.conv5 = FeaStConv(16, 16, heads=heads)
        self.conv6 = FeaStConv(16, 16, heads=heads)
        self.conv7 = FeaStConv(16, 16, heads=heads)
        self.conv8 = FeaStConv(16, 16, heads=heads)
        self.conv9 = FeaStConv(16, 16, heads=heads)
        self.conv10 = FeaStConv(16, 16, heads=heads)
        self.lin1 = Linear(16, 16)
        self.lin2 = Linear(16, 4)
        self.s1 = SELU()
        self.s2 = SELU()
        self.s3 = SELU()
        self.s4 = SELU()
        self.s5 = SELU()
        self.s6 = SELU()
        self.s7 = SELU()
        self.s8 = SELU()
        self.s9 = SELU()
        self.s10 = SELU()
        self.s11 = SELU()
        self.s12 = SELU()
        self.out = Linear(4, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.pre_lin(x) if self.masif_descr else x
        x = self.conv1(x, edge_index)
        x = self.s1(x)
        x = self.conv2(x, edge_index)
        x = self.s2(x)
        x = self.conv3(x, edge_index)
        x = self.s3(x)
        x = self.conv4(x, edge_index)
        x = self.s4(x)
        x = self.conv5(x, edge_index)
        x = self.s5(x)
        x = self.conv6(x, edge_index)
        x = self.s6(x)
        x = self.conv7(x, edge_index)
        x = self.s7(x)
        x = self.conv8(x, edge_index)
        x = self.s8(x)
        x = self.conv9(x, edge_index)
        x = self.s9(x)
        x = self.conv10(x, edge_index)
        x = self.s10(x)
        x = self.lin1(x)
        x = self.s11(x)
        x = self.lin2(x)
        x = self.s12(x)
        x = self.out(x)
        x = torch.sigmoid(x)

        return x


class FourteenConv(torch.nn.Module):
    def __init__(self, n_features, heads=4, masif_descr=False):
        # REMEMBER TO UPDATE MODEL NAME
        super(FourteenConv, self).__init__()
        self.masif_descr = masif_descr
        if masif_descr is True:
            self.pre_lin = Linear(80, n_features)
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        self.conv2 = FeaStConv(16, 16, heads=heads)
        self.conv3 = FeaStConv(16, 16, heads=heads)
        self.conv4 = FeaStConv(16, 16, heads=heads)
        self.conv5 = FeaStConv(16, 16, heads=heads)
        self.conv6 = FeaStConv(16, 16, heads=heads)
        self.conv7 = FeaStConv(16, 16, heads=heads)
        self.conv8 = FeaStConv(16, 16, heads=heads)
        self.conv9 = FeaStConv(16, 16, heads=heads)
        self.conv10 = FeaStConv(16, 16, heads=heads)
        self.conv11 = FeaStConv(16, 16, heads=heads)
        self.conv12 = FeaStConv(16, 16, heads=heads)
        self.conv13 = FeaStConv(16, 16, heads=heads)
        self.conv14 = FeaStConv(16, 16, heads=heads)
        self.s1 = SELU()
        self.s2 = SELU()
        self.s3 = SELU()
        self.s4 = SELU()
        self.s5 = SELU()
        self.s6 = SELU()
        self.s7 = SELU()
        self.s8 = SELU()
        self.s9 = SELU()
        self.s10 = SELU()
        self.s11 = SELU()
        self.s12 = SELU()
        self.s13 = SELU()
        self.s14 = SELU()
        self.s15 = SELU()
        self.s16 = SELU()
        self.lin1 = Linear(16, 16)
        self.lin2 = Linear(16, 4)
        self.out = Linear(4, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.pre_lin(x) if self.masif_descr else x
        x = self.conv1(x, edge_index)
        x = self.s1(x)
        x = self.conv2(x, edge_index)
        x = self.s2(x)
        x = self.conv3(x, edge_index)
        x = self.s3(x)
        x = self.conv4(x, edge_index)
        x = self.s4(x)
        x = self.conv5(x, edge_index)
        x = self.s5(x)
        x = self.conv6(x, edge_index)
        x = self.s6(x)
        x = self.conv7(x, edge_index)
        x = self.s7(x)
        x = self.conv8(x, edge_index)
        x = self.s8(x)
        x = self.conv9(x, edge_index)
        x = self.s9(x)
        x = self.conv10(x, edge_index)
        x = self.s10(x)
        x = self.conv11(x, edge_index)
        x = self.s11(x)
        x = self.conv12(x, edge_index)
        x = self.s12(x)
        x = self.conv13(x, edge_index)
        x = self.s13(x)
        x = self.conv14(x, edge_index)
        x = self.s14(x)
        x = self.lin1(x)
        x = self.s15(x)
        x = self.lin2(x)
        x = self.s16(x)
        x = self.out(x)
        x = torch.sigmoid(x)

        return x

class ThreeConvBlock(torch.nn.Module):
    # Too many parameters?
    def __init__(self, n_features, lin2=4, heads=4):
        super(ThreeConvBlock, self).__init__()
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        self.conv2 = FeaStConv(16, 16, heads=heads)
        self.conv3 = FeaStConv(16, 16, heads=heads)
        self.lin1 = Linear(16, 32)
        self.lin2 = Linear(32, lin2)
        self.out = Linear(lin2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.lin1(x)
        x = x.relu()
        inter = self.lin2(x)
        x = inter.relu()
        x = self.out(x)
        x = torch.sigmoid(x)

        return x, torch.sigmoid(inter)


class ThreeConvBlock2(torch.nn.Module):
    # Too many parameters?
    def __init__(self, n_features, lin2=4, heads=4):
        super(ThreeConvBlock2, self).__init__()
        self.conv1 = FeaStConv(n_features, 32, heads=heads)
        self.conv2 = FeaStConv(32, 32, heads=heads)
        self.conv3 = FeaStConv(32, 32, heads=heads)
        self.lin1 = Linear(32, 64)
        self.lin2 = Linear(64, lin2)
        self.out = Linear(lin2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.lin1(x)
        x = x.relu()
        inter = self.lin2(x)
        x = inter.relu()
        x = self.out(x)
        x = torch.sigmoid(x)

        return x, torch.sigmoid(inter)

class GraphThreeConvBlock(torch.nn.Module):
    # Too many parameters?
    def __init__(self, n_features, lin2=4, heads=4):
        super(GraphThreeConvBlock, self).__init__()
        self.conv1 = GCNConv(n_features, 16, improved=True)
        self.conv2 = GCNConv(16, 16, improved=True)
        self.conv3 = GCNConv(16, 16, improved=True)
        self.lin1 = Linear(16, 32)
        self.lin2 = Linear(32, lin2)
        self.out = Linear(lin2, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        x = self.conv2(x, edge_index)
        x = x.relu()
        x = self.conv3(x, edge_index)
        x = x.relu()
        x = self.lin1(x)
        x = x.relu()
        inter = self.lin2(x)
        x = inter.relu()
        x = self.out(x)
        x = torch.sigmoid(x)

        return x, torch.sigmoid(inter)


class PretrainedBlocks(torch.nn.Module):
    def __init__(self, paths, n_features=4, lin2=4, heads=4):
        super(PretrainedBlocks, self).__init__()
        self.blocks = [ThreeConvBlock(n_features, lin2, heads) for path in paths]
        self.blocks = torch.nn.ModuleList(self.blocks)
        self.batches = [BatchNorm(lin2) for n in range(1, len(paths))]
        self.batches = torch.nn.ModuleList(self.batches)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for block, path in zip(self.blocks, paths):
            block = block.load_state_dict(torch.load(path, map_location=self.device))

    def forward(self, data):
        out = None
        for idx, block in enumerate(self.blocks):
            x = data.x
            out, inter = block(data)
            x += inter
            x = self.batches[idx](x) if idx < len(self.blocks)-1 else x
            data.x = x

        return out


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
        torch.nn.init.kaiming_normal_(self.conv1, nonlinearity='relu')
        self.conv2 = FeaStConv(4, 4, heads=heads)
        torch.nn.init.kaiming_normal_(self.conv2, nonlinearity='relu')
        self.conv3 = FeaStConv(4, 4, heads=heads)
        torch.nn.init.kaiming_normal_(self.conv3, nonlinearity='relu')
        self.conv4 = FeaStConv(4, out_features, heads=heads)
        torch.nn.init.kaiming_normal_(self.conv4, nonlinearity='relu')
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
        x, edge_index, _, _, _, _ = self.pool(x, edge_index)
        x = self.conv4(x, edge_index)
        x = x.relu()
        x = self.batch(x)

        return x, edge_index


class TwentyConvPool(torch.nn.Module):

    def __init__(self, n_features, heads=4):
        super(TwentyConvPool, self).__init__()
        self.block1 = FourConvPoolBlock(n_features, 4, heads=heads)
        self.block2 = FourConvPoolBlock(4, 4, heads=heads)
        self.block3 = FourConvPoolBlock(4, 4, heads=heads)
        self.block4 = FourConvBlock(4, 4, heads=heads)
        self.block5 = FourConvBlock(4, 4, heads=heads)
        self.lin1 = Linear(4, 64)
        self.lin2 = Linear(64, 64)
        self.lin3 = Linear(64, 16)
        self.out = Linear(16, 1)

    def forward(self, data):
        # Should edge_index get updated?
        x1, edge_index = data.x, data.edge_index
        x1, edge_index = self.block1(x1, edge_index)
        x2, edge_index = self.block2(x1, edge_index)
        x3, edge_index = self.block3(x2, edge_index)
        x4 = self.block4(x3, edge_index)
        x5 = self.block5(x4, edge_index)
        z = self.lin1(x5)
        z = z.relu()
        z = self.lin2(z)
        z = z.relu()
        z = self.lin3(z)
        z = z.relu()
        z = self.out(z)
        z = torch.sigmoid(z)

        return z


class EightConv(torch.nn.Module): # Model 19

    def __init__(self, n_features, heads=4):
        super(EightConv, self).__init__()
        self.block1 = FourConvBlock(4, 4)
        self.block2 = FourConvBlock(4, 4)
        self.lin1 = Linear(4, 64)
        self.lin2 = Linear(64, 64)
        self.lin3 = Linear(64, 16)
        self.out = Linear(16, 1)

    def forward(self, data):
        z, edge_index = data.x, data.edge_index
        z = self.block1(z, edge_index)
        z = self.block2(z, edge_index)
        z = self.lin1(z)
        z = z.relu()
        z = self.lin2(z)
        z = z.relu()
        z = self.lin3(z)
        z = z.relu()
        z = self.out(z)
        z = torch.sigmoid(z)

        return z


class TwentyConvNoRes(torch.nn.Module):
    # Try again with much higher learning rate!
    def __init__(self, n_features, heads=4):
        super(TwentyConvNoRes, self).__init__()
        self.block1 = FourConvBlock(n_features, 4, heads=heads)
        self.block2 = FourConvBlock(4, 4, heads=heads)
        self.block3 = FourConvBlock(4, 4, heads=heads)
        self.block4 = FourConvBlock(4, 4, heads=heads)
        self.block5 = FourConvBlock(4, 4, heads=heads)
        self.lin1 = Linear(4, 64)
        torch.nn.init.kaiming_normal_(self.lin1, nonlinearity='relu')
        self.lin2 = Linear(64, 64)
        torch.nn.init.kaiming_normal_(self.lin2, nonlinearity='relu')
        self.lin3 = Linear(64, 16)
        torch.nn.init.kaiming_normal_(self.lin3, nonlinearity='relu')
        self.out = Linear(16, 1)
        torch.nn.init.kaiming_normal_(self.out, nonlinearity='relu')

    def forward(self, data):
        # Should edge_index get updated?
        x1, edge_index = data.x, data.edge_index
        x1 = self.block1(x1, edge_index)
        x2 = self.block2(x1, edge_index)
        x3 = self.block3(x2, edge_index)
        x4 = self.block4(x3, edge_index)
        x5 = self.block5(x4, edge_index)
        z = self.lin1(x5)
        z = z.relu()
        z = self.lin2(z)
        z = z.relu()
        z = self.lin3(z)
        z = z.relu()
        z = self.out(z)
        z = torch.sigmoid(z)

        return z


class MultiScaleFeaStNet(torch.nn.Module):
    def __init__(self, n_features, heads=4):
        super(MultiScaleFeaStNet,self).__init__()
        self.conv1 = FeaStConv(n_features, 16, heads=heads)
        self.conv2 = FeaStConv(16, 32, heads=heads)
        self.conv3 = FeaStConv(32, 64, heads=heads)
        self.conv4 = FeaStConv(64, 32, heads=heads)
        self.conv5 = FeaStConv(64, 16, heads=heads)
        self.lin1 = Linear(32, 256)
        self.lin2 = Linear(256, 6890)
        self.out = Linear(6890, 1)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = x.relu()
        cluster1 = graclus(edge_index, num_nodes=x.shape[0])
        pooled_1 = data
        pooled_1.x = x
        pooled_1 = max_pool(cluster1, pooled_1)
        edge_index_2 = pooled_1.edge_index
        x2 = pooled_1.x
        x2 = self.conv2(x2, edge_index_2)
        x2 = x2.relu()
        cluster2 = graclus(edge_index_2, num_nodes=x2.shape[0])
        pooled_2 = pooled_1
        pooled_2.x = x2
        pooled_2 = max_pool(cluster2, pooled_2)
        edge_index_3 = pooled_2.edge_index
        x3 = pooled_2.x
        x3 = self.conv3(x3, edge_index_3)
        x3 = x3.relu()
        x3 = self.conv4(x3, edge_index_3)
        x3 = x3.relu()
        x3 = knn_interpolate(x3, pooled_2.pos, pooled_1.pos)
        x3 = torch.cat((x2, x3), dim=1)
        x3 = self.conv5(x3, edge_index_2)
        x3 = x3.relu()
        x3 = knn_interpolate(x3, pooled_1.pos, data.pos)
        x = torch.cat((x, x3), dim=1)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = torch.sigmoid(self.out(x))

        return x


class SageNet(torch.nn.Module):
    def __init__(self, n_features):
        super(SageNet, self).__init__()
        self.conv1 = SAGEConv(n_features, 16, normalize=False)
        self.conv2 = SAGEConv(16, 16, normalize=False)
        self.conv3 = SAGEConv(16, 16, normalize=False)
        self.conv4 = SAGEConv(16, 16, normalize=False)
        self.conv5 = SAGEConv(16, 16, normalize=False)
        self.conv6 = SAGEConv(16, 16, normalize=False)
        self.conv7 = SAGEConv(16, 16, normalize=False)
        self.conv8 = SAGEConv(16, 16, normalize=False)
        self.conv9 = SAGEConv(16, 16, normalize=False)
        self.lin1 = Linear(16, 64)
        self.lin2 = Linear(64, 16)
        self.out = Linear(16, 1)
        self.s1 = SELU()
        self.s2 = SELU()
        self.s3 = SELU()
        self.s4 = SELU()
        self.s5 = SELU()
        self.s6 = SELU()
        self.s7 = SELU()
        self.s8 = SELU()
        self.s9 = SELU()
        self.s10 = SELU()
        self.s11 = SELU()

    def forward(self, x, data_flow):
        data = data_flow[0]
        x = x[data.n_id]
        x = self.s1(self.conv1((x, None), data.edge_index, size=data.size))
        data = data_flow[1]
        x = self.s2(self.conv2((x, None), data.edge_index, size=data.size))
        data = data_flow[2]
        x = self.s3(self.conv3((x, None), data.edge_index, size=data.size))
        data = data_flow[3]
        x = self.s4(self.conv4((x, None), data.edge_index, size=data.size))
        data = data_flow[4]
        x = self.s5(self.conv5((x, None), data.edge_index, size=data.size))
        data = data_flow[5]
        x = self.s6(self.conv6((x, None), data.edge_index, size=data.size))
        data = data_flow[6]
        x = self.s7(self.conv7((x, None), data.edge_index, size=data.size))
        data = data_flow[7]
        x = self.s8(self.conv8((x, None), data.edge_index, size=data.size))
        data = data_flow[8]
        x = self.s9(self.conv9((x, None), data.edge_index, size=data.size))
        x = self.s10(self.lin1(x))
        x = self.s11(self.lin2(x))
        return self.out(x)
