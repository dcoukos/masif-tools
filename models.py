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


class GraphUNet(torch.nn.Module):
    '''
    Adapted from Pytorch_geometrics's GraphUNet. Modified to use FeaStNet's convolutional function

     adapted Multi-scale model from Verma et al. and the Graph U-Net model
    from the `"Graph U-Nets"
    <https://arxiv.org/abs/1905.05178>`_ paper which implements a U-Net like
    architecture with graph pooling and unpooling operations.

    Args:
        in_channels (int): Size of each input sample.
        hidden_channels (int): Size of each hidden sample.
        out_channels (int): Size of each output sample.
        depth (int): The depth of the U-Net architecture.
        pool_ratios (float or [float], optional): Graph pooling ratio for each
            depth. (default: :obj:`0.5`)
        sum_res (bool, optional): If set to :obj:`False`, will use
            concatenation for integration of skip connections instead
            summation. (default: :obj:`True`)
        act (torch.nn.functional, optional): The nonlinearity to use.
            (default: :obj:`torch.nn.functional.relu`)
    '''

    def __init__(self, in_channels, hidden_channels, out_channels, depth,
                 pool_ratios=0.5, sum_res=True, act=F.relu):
        super(GraphUNet, self).__init__()
        assert depth >= 1
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.depth = depth
        self.pool_ratios = repeat(pool_ratios, depth)
        self.act = act
        self.sum_res = sum_res

        channels = hidden_channels

        self.down_convs = torch.nn.ModuleList()
        self.pools = torch.nn.ModuleList()
        self.down_convs.append(FeaStConv(in_channels, channels))
        for i in range(depth):
            self.pools.append(TopKPooling(channels, self.pool_ratios[i]))
            self.down_convs.append(FeaStConv(channels, channels))

        in_channels = channels if sum_res else 2 * channels

        self.up_convs = torch.nn.ModuleList()
        for i in range(depth - 1):
            self.up_convs.append(FeaStConv(in_channels, channels))
        self.up_convs.append(FeaStConv(in_channels, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        for conv in self.down_convs:
            conv.reset_parameters()
        for pool in self.pools:
            pool.reset_parameters()
        for conv in self.up_convs:
            conv.reset_parameters()

    def forward(self, x, edge_index, labels, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        edge_weight = x.new_ones(edge_index.size(1))

        x = self.down_convs[0](x, edge_index)
        x = self.act(x)

        xs = [x]
        edge_indices = [edge_index]
        edge_weights = [edge_weight]
        perms = []

        for i in range(1, self.depth + 1):
            edge_index, edge_weight = self.augment_adj(edge_index, edge_weight,
                                                       x.size(0))
            x, edge_index, edge_weight, batch, perm, _ = self.pools[i - 1](
                x, edge_index, edge_weight, batch)

            x = self.down_convs[i](x, edge_index)
            x = self.act(x)

            if i < self.depth:
                xs += [x]
                edge_indices += [edge_index]
                edge_weights += [edge_weight]
            perms += [perm]

        for i in range(self.depth):
            j = self.depth - 1 - i

            res = xs[j]
            edge_index = edge_indices[j]
            edge_weight = edge_weights[j]
            perm = perms[j]

            up = torch.zeros_like(res)
            up[perm] = x
            x = res + up if self.sum_res else torch.cat((res, up), dim=-1)

            x = self.up_convs[i](x, edge_index, edge_weight)
            x = self.act(x) if i < self.depth - 1 else x

        loss = F.binary_cross_entropy(x, target=labels, weight=generate_weights(labels))
        return loss, x

        def augment_adj(self, edge_index, edge_weight, num_nodes):
            edge_index, edge_weight = add_self_loops(edge_index, edge_weight,
                                                     num_nodes=num_nodes)
            edge_index, edge_weight = sort_edge_index(edge_index, edge_weight,
                                                      num_nodes)
            edge_index, edge_weight = spspmm(edge_index, edge_weight, edge_index,
                                             edge_weight, num_nodes, num_nodes,
                                             num_nodes)
            edge_index, edge_weight = remove_self_loops(edge_index, edge_weight)
            return edge_index, edge_weight

        def __repr__(self):
            return '{}({}, {}, {}, depth={}, pool_ratios={})'.format(
                self.__class__.__name__, self.in_channels, self.hidden_channels,
                self.out_channels, self.depth, self.pool_ratios)
