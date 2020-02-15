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
