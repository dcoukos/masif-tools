import torch
import torch.nn.functional as F
from torch.nn import Linear, Dropout, Sequential, ReLU
from torch_geometric.nn import GCNConv, FeaStConv, MessagePassing, knn_graph, BatchNorm, TopKPooling, graclus, max_pool
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


class FourConv(torch.nn.Module): # Model 18

    def __init__(self, n_features, heads=4):
        super(FourConv, self).__init__()
        self.block1 = FourConvBlock(4, 4)
        self.lin1 = Linear(4, 64)
        self.lin2 = Linear(64, 64)
        self.lin3 = Linear(64, 16)
        self.out = Linear(16, 1)

    def forward(self, data):
        z, edge_index = data.x, data.edge_index
        z = self.block1(z, edge_index)
        z = self.lin1(z)
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
        x2 = max_pool(cluster1, x)
        edge_index_2 = x2.edge_index
        x2 = x2.x
        x2 = self.conv2(x2, edge_index2)
        x2 = x2.relu()
        cluster2 = graclus(edge_index_2, num_nodes=x2.shape[0])
        x3 = max_pool(cluster2, x2)
        edge_index_3 = x3.edge_index
        x3 = x3.x
        x3 = self.conv3(x3, edge_index_3)
        x3 = x3.relu()
        x3 = self.conv4(x3, edge_index_3)
        x3 = x3.relu()
        x3 = x3[cluster2]
        x3 = torch.cat((x2, x3), dim=1)
        x3 = self.conv5(x3, edge_index_2)
        x3 = x3.relu()
        x3 = x3[cluster1]
        x = torch.cat((x, x3), dim=1)
        x = self.lin1(x)
        x = x.relu()
        x = self.lin2(x)
        x = x.relu()
        x = torch.sigmoid(self.out(x))

        return x
