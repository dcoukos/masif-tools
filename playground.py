from utils import generate_surface
import params as p

generate_surface(p.model_type, './models/Feb13_11:47_16b/best.pt', '3BIK_A', False)

generate_surface(p.model_type, './models/Feb13_11:47_16b/best.pt', '4ZQK_A', False)


import glob
paths = glob.glob(os.path.expanduser('~/Desktop/Drawer/LPDI/masif-tools/structures/*'))


test_paths = []
with open('./lists/testing.txt', 'r') as f:
    for line in f:
        line = line.split('\n')[0]
        test_paths.append(line)
test_paths
for name in test_paths:
    if len(name.split('_')) == 3:
        a, b, c = name.split('_')
        name = a + '_' + b
        structure2 = a + '_' + c
        test_paths.append(structure2)
    new_path = os.path.expanduser('~/Desktop/Drawer/LPDI/masif-tools/structures/test/') + name + \
        '.ply'
    path = os.path.expanduser('~/Desktop/Drawer/LPDI/masif-tools/structures/') + name + \
        '.ply'
    try:
        os.replace(path, new_path)
    except:
        print('{} not found'.format(name))

# rest done from command line.

import pathlib
pathlib.Path().absolute()



#Using the edge_index
import torch
from torch_geometric.transforms import FaceToEdge, PointPairFeatures
from dataset import MiniStructures
import pandas as pd
import numpy as np
from torch_geometric.data import DataLoader
dataset = MiniStructures(root='./datasets/mini/', pre_transform=FaceToEdge(), transform=PointPairFeatures())

dataset = MiniStructures()
dataloader = DataLoader(dataset, batch_size=20)
iterator = iter(dataloader)
batch = next(iterator)
batch.to_data_list()

dataset[0]
dataset[0].x.shape
edges = dataset[0].edge_index.t()
edges_df = pd.DataFrame(edges.numpy(), columns=['origin', 'destination'])

origins = torch.tensor(edges_df.origin.unique())
# max number of neighbors = 13
neighbors_per_node = []
for origin in edges_df.origin.unique():
    neighbors_per_node.append(torch.tensor(edges_df[edges_df.origin == origin].destination.values))
neighbors = torch.nn.utils.rnn.pad_sequence(neighbors_per_node, batch_first=True,
                                            padding_value=1e10)
# replace 0 w/ None
grouped_edges = torch.cat((origins.reshape(-1,1), neighbors), dim=1)
# Is concatenating necessary?
grouped_edges

for i, row in enumerate(neighbors):
    for j in row[row != 1e10]:
        print position

i, row = next(enumerate(neighbors))
j = next(iter(row[row!= 1e10]))
j = row[row != 1e10]
norms = dataset[0].norm[j]
norms
delta_n = n_i - norms
n_i
n_i = dataset[0].norm[i]
n_j = dataset[0].norm[j]
p_i = dataset[0].pos[i]
p_j = dataset[0].pos[j]
delta_p = p_i - p_j

k_ij = (delta_n*delta_p)/delta_p.norm()**2
k_ij
dataset

# Applying pretransformation once and saving datasets!
# New strategy... load the data first, without pretransforms,
# then load into a dataloader and batch apply the transforms.


# Why is it not using gpu?
from utils import apply_pretransforms
from dataset import Structures_SI_mem, Structures
from torch_geometric.transforms import *
from transforms import *

dataset = Structures_SI(root='./datasets/thous_train/', pre_transform=Compose((Center(), FaceAttributes(),
                                               NodeCurvature(), FaceToEdge(),
                                               TwoHop(), AddShapeIndex())))
dataset = Structures_SI_mem(root='./datasets/thous_train/',
                        pre_transform=Compose((Center(), FaceAttributes(),
                                               NodeCurvature(), FaceToEdge(),
                                               TwoHop(), AddShapeIndex())))

dataset[0]


dataset[0]

apply_pretransforms()

# Checking why my transformations keep crashing...
import torch
from torch_geometric.transforms import Compose, FaceToEdge, TwoHop, Center
from transforms import *
import params as p
from dataset import Structures

dataset = Structures(pre_transform=Compose((Center(), FaceAttributes(), NodeCurvature(), FaceToEdge(), TwoHop())))

data = Structures(root='./datasets/thous/')[234]
pre_transform = Compose((Center(), FaceAttributes(), NodeCurvature(), FaceToEdge(), TwoHop()))
data1 = dataset[233]
data2 = dataset[234]
data3 = dataset[235]
data2


data1 = pre_transform(data1)
data2 = pre_transform(data2)
data3 = pre_transform(data3)

data2
# For normal projections.


# Visualize shape data:
structure = Structures(root='./datasets/thous_train', pre_transform=Compose((Center(), FaceAttributes(), NodeCurvature(), FaceToEdge(), TwoHop())))[457]
set = Structures(root='./datasets/thous/')
converter = Compose((Center(), FaceAttributes(), NodeCurvature(), FaceToEdge(remove_faces=False), TwoHop()))
structure = converter(set[457])

structure.y
save_ply(
    filename='example_curvature.ply',
    vertices=structure.pos.detach().numpy(),
    normals=structure.norm.detach().numpy(),
    faces=structure.face.t().detach().numpy(),
    charges=structure.x[:, 0].reshape(-1, 1).detach().numpy(),
    hbond=structure.shape_index.reshape(-1, 1).detach().numpy(),
    hphob=structure.x[:, 1].reshape(-1, 1).detach().numpy(),
    iface=structure.y.detach().numpy()
)
from dataset import Structures
from utils import has_nan

datalist = Structures('./datasets/thous_train/')
_, idx = has_nan(datalist)
datalist = [datalist[i] for i in range(0, len(datalist)) if i not in idx]
len(datalist)

def save_ply(
    filename,
    vertices,
    faces=[],
    normals=None,
    charges=None,
    vertex_cb=None,
    hbond=None,
    hphob=None,
    iface=None,
    normalize_charges=False,
):
    """ Pablo Gainza - LPDI STI EPFL 2019
        Released under an Apache License 2.0

        Save vertices, mesh in ply format.
        vertices: coordinates of vertices
        faces: mesh
    """
    import pymesh  # No pymesh on gpu cluster
    mesh = pymesh.form_mesh(vertices, faces)
    if normals is not None:
        n1 = normals[:, 0]
        n2 = normals[:, 1]
        n3 = normals[:, 2]
        mesh.add_attribute("vertex_nx")
        mesh.set_attribute("vertex_nx", n1)
        mesh.add_attribute("vertex_ny")
        mesh.set_attribute("vertex_ny", n2)
        mesh.add_attribute("vertex_nz")
        mesh.set_attribute("vertex_nz", n3)
    if charges is not None:
        mesh.add_attribute("charge")
        if normalize_charges:
            charges = charges / 10
        mesh.set_attribute("charge", charges)
    if hbond is not None:
        mesh.add_attribute("hbond")
        mesh.set_attribute("hbond", hbond)
    if vertex_cb is not None:
        mesh.add_attribute("vertex_cb")
        mesh.set_attribute("vertex_cb", vertex_cb)
    if hphob is not None:
        mesh.add_attribute("vertex_hphob")
        mesh.set_attribute("vertex_hphob", hphob)
    if iface is not None:
        mesh.add_attribute("vertex_iface")
        mesh.set_attribute("vertex_iface", iface)

    pymesh.save_mesh(
        filename, mesh, *mesh.get_attribute_names(), use_float=True, ascii=True
    )


# ------ Adding shape index features to full dataset ---------------------
import torch
from dataset import Structures, StructuresDataset
from transforms import *
from torch_geometric.transforms import FaceToEdge, TwoHop, RandomRotate, Compose, Center
import params as p
from tqdm import tqdm
from torch_geometric.data import DataLoader
import pandas as pd
import numpy as np


test = Structures(root='./datasets/masif_site_test/',
                         pre_transform=Compose((Center(), FaceAttributes(),
                         NodeCurvature(), FaceToEdge(),
                         TwoHop())))
train = Structures(root='./datasets/masif_site_train/',
                   pre_transform=Compose((Center(), FaceAttributes(),
                                          NodeCurvature(), FaceToEdge(),
                                          TwoHop())))


# ---------------- Trying to generate numpy files ----------------------


train_structures = Structures(root='./datasets/{}_train'.format(p.dataset))


collection = []
for data in tqdm(test):
    collection.append(pd.DataFrame(torch.cat((data.pos, data.norm, data.x,
                                              #data.shape_index.view(-1,1),
                                              data.y), dim=1).numpy(),
                      columns=['x', 'y', 'z', 'norm_x', 'norm_y', 'norm_z',
                               'charge', 'hbond', 'hphob', 'shape_index',
                               'interface']))
collection[1]

test_array = pd.DataFrame(collection, columns=['structure'])


collection = []
for data in tqdm(train):
    collection.append(pd.DataFrame(torch.cat((data.pos, data.norm, data.x,
                                              #data.shape_index.view(-1,1),
                                              data.y), dim=1).numpy(),
                      columns=['x', 'y', 'z', 'norm_x', 'norm_y', 'norm_z',
                               'charge', 'hbond', 'hphob', 'shape_index',
                               'interface']))
train_df = pd.DataFrame(collection, columns=['structure'])

names = np.array(torch.load('./datasets/full_train_ds/raw/full_indices.pt'))[:,1]
train_df['pdb_id'] = names
torch.save(train_df, './datasets/full_train_ds_numpy.pt')


# ------------------------ Find the ROC AUC for PDL1 -----------------------------
from dataset import read_ply
import torch
import os
import params as p
from glob import glob
from torch_geometric.transforms import Compose, FaceToEdge, TwoHop, Center
from transforms import *
from models import *
import datetime
import pathlib
from tqdm import tqdm
from utils import save_ply
from sklearn.metrics import roc_auc_score


converter = Compose((Center(), FaceAttributes(),
                     NodeCurvature(), FaceToEdge(),
                     TwoHop(), AddShapeIndex()))
pdb_code = '3BIK_A'
code2 = '4ZQK_A'
path = glob('./structures/test/{}.ply'.format(code2))[0]
name = path.split('/')[-1]
structure = read_ply(path)

face = structure.face
structure = converter(structure)

device = torch.device('cpu')
structure.x.shape[1]

model0 = ThreeConvBlock(4, 4, 4)
model1 = ThreeConvBlock(4, 4, 4)
model2 = ThreeConvBlock(4, 4, 4)

model0.load_state_dict(torch.load('./models/Feb16_14:09_20b/best_0.pt', map_location=device))
model1.load_state_dict(torch.load('./models/Feb16_14:09_20b/best_1.pt', map_location=device))
model2.load_state_dict(torch.load('./models/Feb16_14:09_20b/best_2.pt', map_location=device))

model0.eval()
model1.eval()
model2.eval()

prediction0, inter0 = model0(structure)
structure.x += inter0
prediction1, inter1 = model1(structure)
structure.x += inter1
prediction2, inter2 = model2(structure)




path0 = os.path.expanduser('~/Desktop/Drawer/LPDI/masif-tools/surfaces/Feb16_14:09_20b/best0/3BIK_A.ply')
path1 = os.path.expanduser('~/Desktop/Drawer/LPDI/masif-tools/surfaces/Feb16_14:09_20b/best1/3BIK_A.ply')
path2 = os.path.expanduser('~/Desktop/Drawer/LPDI/masif-tools/surfaces/Feb16_14:09_20b/best2/3BIK_A.ply')




roc_auc_score(structure.y, prediction2.detach())
roc_auc_score(structure.y, prediction2.detach())

structure.pos.shape
structure.norm.shape
face.t().shape
structure.x.shape
prediction0.shape

save_ply(
    filename=path0,
    vertices=structure.pos.detach().numpy(),
    normals=structure.norm.detach().numpy(),
    faces=face.t().detach().numpy(),
    charges=structure.x[:, 0].reshape(-1, 1).detach().numpy(),
    hbond=structure.x[:, 3].reshape(-1, 1).detach().numpy(),
    hphob=structure.x[:, 2].reshape(-1, 1).detach().numpy(),
    iface=prediction0.detach().numpy()
)




# ---------- Creating out of memory Datasets ------------------------


train[0]
train[0]

loader = DataLoader(train)

for l in loader:
    l.y = l.x

train




# ------------- Trying to understand neighborhood sampling ---------------------
import os.path as osp

import argparse
import torch
import torch.nn.functional as F
from torch_geometric.datasets import Reddit
from torch_geometric.data import NeighborSampler
from torch_geometric.nn import SAGEConv, GATConv
from dataset import Structures

parser = argparse.ArgumentParser()
#parser.add_argument('--model', type=str, default='SAGE')
args = parser.parse_args()
assert args.model in ['SAGE', 'GAT']


data = Structures()
loader = NeighborSampler(data, size=[25, 10], num_hops=2, batch_size=1000,
                         shuffle=True, add_self_loops=True)


class SAGENet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SAGENet, self).__init__()
        self.conv1 = SAGEConv(in_channels, 16, normalize=False)
        self.conv2 = SAGEConv(16, out_channels, normalize=False)

    def forward(self, x, data_flow):
        data = data_flow[0]
        x = x[data.n_id]
        x = F.relu(self.conv1((x, None), data.edge_index, size=data.size))
        x = F.dropout(x, p=0.5, training=self.training)
        data = data_flow[1]
        x = self.conv2((x, None), data.edge_index, size=data.size)
        return F.log_softmax(x, dim=1)


class GATNet(torch.nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GATNet, self).__init__()
        self.conv1 = GATConv(in_channels, 8, heads=8, dropout=0.6)
        self.conv2 = GATConv(8 * 8, out_channels, heads=1, concat=True,
                             dropout=0.6)

    def forward(self, x, data_flow):
        block = data_flow[0]
        x = x[block.n_id]
        x = F.elu(
            self.conv1((x, x[block.res_n_id]), block.edge_index,
                       size=block.size))
        x = F.dropout(x, p=0.6, training=self.training)
        block = data_flow[1]
        x = self.conv2((x, x[block.res_n_id]), block.edge_index,
                       size=block.size)
        return F.log_softmax(x, dim=1)


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
Net = SAGENet if args.model == 'SAGE' else GATNet
model = Net(dataset.num_features, dataset.num_classes).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)


def train():
    model.train()

    total_loss = 0
    for data_flow in loader(data.train_mask):
        optimizer.zero_grad()
        out = model(data.x.to(device), data_flow.to(device))
        loss = F.nll_loss(out, data.y[data_flow.n_id].to(device))
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * data_flow.batch_size
    return total_loss / data.train_mask.sum().item()


def test(mask):
    model.eval()

    correct = 0
    for data_flow in loader(mask):
        pred = model(data.x.to(device), data_flow.to(device)).max(1)[1]
        correct += pred.eq(data.y[data_flow.n_id].to(device)).sum().item()
    return correct / mask.sum().item()


for epoch in range(1, 31):
    loss = train()
    test_acc = test(data.test_mask)
    print('Epoch: {:02d}, Loss: {:.4f}, Test: {:.4f}'.format(
        epoch, loss, test_acc))

# ---------------------- Trying to use datastructures ----------------------------
import torch
from dataset import StructuresDataset
from transforms import *
from torch_geometric.transforms import *
from models import TwoConv
import params as p

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
cpu = torch.device('cpu')
# reproducibility
torch.manual_seed(p.random_seed)
np.random.seed(p.random_seed)
learn_rate = p.learn_rate


model = TwoConv(3, heads=p.heads).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=p.weight_decay)

trainset = StructuresDataset(root='./datasets/full_train_ds/',
                             pre_transform=Compose((FaceAttributes(), NodeCurvature(),
                                                    FaceToEdge(), TwoHop())))


validset = StructuresDataset(root='./datasets/full_test_ds',
                             pre_transform=Compose((FaceAttributes(), NodeCurvature(),
                                                    FaceToEdge(), TwoHop())))

samples = len(trainset)
cutoff = int(np.floor(samples*(1-p.validation_split)))
train_indices = torch.tensor([i for i in range(0, cutoff)])
train = trainset[train_indices]

validset = trainset[cutoff:]
trainset = trainset[:cutoff]

sorted(glob.glob('./datasets/full_train_ds/processed/data_*.pt'))



import itertools
x = itertools.product([11, 12], [1,23], )
for prd in x:
    print(prd)

test = StructuresDataset(root='./datasets/named_masif_test_ds/',
                         pre_transform=Compose((FaceToEdge(), TwoHop(), AddMasifDescriptor(True))),
                         prefilter=None)

train = StructuresDataset(root='./datasets/named_masif_train_ds/',
                         pre_transform=Compose((FaceToEdge(), TwoHop(), AddMasifDescriptor(True))),
                         prefilter=None)



# ----------- Trying to initialize layers of deep networks with previously learned ones ----------

import torch
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.transforms import FaceToEdge, TwoHop, RandomRotate, Compose, Center
from torch_geometric.nn import DataParallel
from dataset import Structures
from transforms import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from utils import generate_weights, generate_example_surfaces, make_model_directory
import params as p
from statistics import mean
import torch.nn.functional as F
from tqdm import tqdm

device = torch.device('cpu')
prev_model = torch.load('./models/Feb27_11:07_exp1_10conv-elec+SI/best.pt', map_location=cpu)
model = p.model_type(5, heads=p.heads).to(cpu)

conv1_weights = prev_model['conv1.weight']
extra_row = torch.ones(1, 64)
conv1_weights = torch.cat((conv1_weights, extra_row), dim=0)
prev_model['conv1.weight'] = conv1_weights

conv1_u = prev_model['conv1.u']
conv1_u = torch.cat((conv1_u, torch.tensor([0.05, 0.05, 0.05, 0.05]).view(1,4)), dim=0)
prev_model['conv1.u'] = conv1_u


model.load_state_dict(prev_model)


# Code for retraining model with random features
conv1_weights = prev_model['conv1.weight']
extra_row = torch.ones(1, 64)*10e-8
conv1_weights = torch.cat((conv1_weights, extra_row), dim=0)
prev_model['conv1.weight'] = conv1_weights

conv1_u = prev_model['conv1.u']
conv1_u = torch.cat((conv1_u, torch.tensor([0.05, 0.05, 0.05, 0.05]).view(1, 4)), dim=0)
prev_model['conv1.u'] = conv1_u
