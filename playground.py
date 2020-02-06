from utils import generate_surface
import params as p

generate_surface(p.model_type, 'models/Jan23_14:40_15b/final.pt', '3BIK_A', False)

generate_surface(p.model_type, 'models/Jan23_14:40_15b/final.pt', '4ZQK_A', False)


import os

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
dataset = MiniStructures(root='./datasets/mini/', pre_transform=FaceToEdge(), transform=PointPairFeatures())
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
from dataset import Structures
import params as p
from transforms import *
from torch_geometric.transforms import Center, Compose, FaceToEdge, TwoHop
import torch

converter = TwoHop()
trainset = Structures(root='./datasets/{}/'.format(p.dataset),
                      pre_transform=Compose((Center(), FaceAttributes(),
                                             NodeCurvature(), FaceToEdge())),
                      transform=converter, prefix=p.dataset)





 # For normal projections.
'''    NOT DOING PROJECTION INTO THE PLANE OF THE NORMAL.
rand = torch.zeros(norms.shape).random_()
rand_norm = rand.norm(dim=1).view(-1, 1)
rand = torch.div(rand, rand_norm)

# PROBLEM W/ DEFINITION
# project to normals
projected = ((rand*norms).sum(-1)/(norms*norms).sum(-1)).view(-1,1)*norms
# take the difference.
e0 = rand - projected
e0 = e0.div(e0.norm(dim=1).view(-1,1))
# take the crossproduct of the two vectors.
e1 = torch.cross(norms, e0, dim=1)
# normalize.
e0 = e0.div(e0.norm(dim=1).view(-1,1))
e1 = e1.div(e1.norm(dim=1).view(-1,1))
# Collect the (indices) of the faces adjacent to the node
'''
