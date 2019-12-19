from glob import glob
from tqdm import tqdm
import torch
from plyfile import PlyData
from torch_geometric.data import Data, InMemoryDataset
from itertools import product


'''
File to generate the dataset from the ply files.

TODO:
    - Create in-memory dataset
    - Create mini in-memory dataset
'''


def convert_data(path_to_raw='./structures/'):
    '''Generate raw unprocessed torch file to generate pyg datasets.
    '''
    # test with 50 structures!

    structures = [read_ply(path) for path in tqdm(glob(path_to_raw+'*'), desc='Reading structures')]
    print('Saving structures to file as pytorch object...')
    torch.save(structures, 'datasets/raw/structures.pt')
    print('Done.')


def convert_mini_data(path_to_raw='./structures/'):
    '''Generate raw unprocessed torch file to generate pyg datasets with fewer
        candidates.
    '''
    # Does this require a different dataset directory? Can try, just back up
    # structures.pt file.

    structures = [read_ply(path) for path in tqdm(glob(path_to_raw+'*')[:200],desc='Reading structures')]
    print('Saving structures to file as pytorch object ...')
    torch.save(structures, 'datasets/raw/mini_structures.pt')
    print('Done.')


def collate(data_list):
    r"""Collates a python list of data objects to the internal storage
    format of :class:`torch_geometric.data.InMemoryDataset`."""
    keys = data_list[0].keys
    data = data_list[0].__class__()

    for key in keys:
        data[key] = []
    slices = {key: [0] for key in keys}

    for item, key in product(data_list, keys):
        data[key].append(item[key])
        if torch.is_tensor(item[key]):
            s = slices[key][-1] + item[key].size(
                item.__cat_dim__(key, item[key]))
        else:
            s = slices[key][-1] + 1
        slices[key].append(s)

    if hasattr(data_list[0], '__num_nodes__'):
        data.__num_nodes__ = []
        for item in data_list:
            data.__num_nodes__.append(item.num_nodes)

    for key in keys:
        item = data_list[0][key]
        if torch.is_tensor(item):
            data[key] = torch.cat(
                data[key], dim=data.__cat_dim__(key, data_list[0][key]))
        elif isinstance(item, int) or isinstance(item, float):
            data[key] = torch.tensor(data[key])

        slices[key] = torch.tensor(slices[key], dtype=torch.long)

    return data, slices


def generate_raw_data():
    data_list = [read_ply(path, True) for path in
                 glob.glob('./temp/*')]
    data, slices = collate(data_list)
    torch.save((data, slices), './structures/raw/data.pt')


def generate_raw_mini_data(n_structures=300):
    data_list = [read_ply(path, True) for path in
                 glob.glob('./temp/*')[:n_structures]]
    data, slices = collate(data_list)
    torch.save((data, slices), './mini_structures/raw/data.pt')


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

    norm = ([torch.tensor(data['vertex'][axis])
            for axis in ['nx', 'ny', 'nz']])
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


"""
class MiniStructures(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None, n=500):
        '''
            Interesting transforms to consider:
                - FaceToEdge --> necessary
                - Change coordinate system to local system to use for training
        '''
        self.n_structures = n
        super(MiniStructures, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property  # What does the property decorator do?
    def raw_file_names(self):
        # Returns empty list as raw copy is local.
        return ['1A0G_A.ply', '1A0G_B.ply']

    @property
    def processed_file_names(self):
        # I don't undesrtand the error this causes. Isn't it supposed to check
        _if_ there are files there and if not make them?,,,
        return []

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.

        data_list = [read_ply(filepath) for filepath in self.raw_file_names]

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
        # Data is saved as one big data structure through collation. Slices are
        # used to reconstruct the original objects.
"""


class Structures(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super(Structures, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        # Returns empty list as raw copy is local.
        return ['structures.pt']

    @property
    def processed_file_names(self):
        # How to know which one is returned as dataset?
        return ['structures.pt']

    def download(self):
        pass

    def process(self):
        # Read data into huge `Data` list.
        data_list = torch.load(self.raw_paths[0])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
