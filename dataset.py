import os
import glob
import torch
import torch_geometric as tg
from plyfile import PlyData
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm
from pandas import DataFrame
'''
File to generate the dataset from the ply files.

TODO:
    - Create in-memory dataset
    - Create mini in-memory dataset
'''


def read_ply(path, learn_iface=True):
    '''
        read_ply from pytorch_geometric does not capture the properties in ply
        file. This function adds to pyg's read_ply function by capturing extra
        properties: charge, hbond, hphob, and iface.

    '''
    with open(path, 'rb') as f:
        data = PlyData.read(f)

    pos = ([torch.tensor(data['vertex'][axis]) for axis in ['x', 'y', 'z']])
    pos = torch.stack(pos, dim=-1)

    norm = ([torch.tensor(data['vertex'][axis]) for axis in ['nx', 'ny', 'nz']])
    norm = torch.stack(norm, dim=-1)

    y = None
    if learn_iface:
        x = ([torch.tensor(data['vertex'][axis]) for axis in
             ['charge', 'hbond', 'hphob']])
        x = torch.stack(x, dim=-1)  # TODO: what does this do again?
        y = [torch.tensor(data['vertex']['iface'])]
        y = torch.stack(y, dim=-1)
    else:
        x = ([torch.tensor(data['vertex'][axis]) for axis in
             ['charge', 'hbond', 'hphob', 'iface']])
        x = torch.stack(x, dim=-1)

    face = None
    if 'face' in data:
        faces = data['face']['vertex_indices']
        faces = [torch.tensor(fa, dtype=torch.long) for fa in faces]
        face = torch.stack(faces, dim=-1)

    data = Data(x=x, pos=pos, face=face, norm=norm, y=y)

    return data
