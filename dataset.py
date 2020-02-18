from glob import glob
from tqdm import tqdm
import torch
from plyfile import PlyData
from torch_geometric.data import Data, InMemoryDataset, Dataset
from itertools import product
from tqdm import tqdm
import params as p
import os.path as osp
import numpy as np
'''
File to generate the dataset from the ply files.

'''
def convert_data(path_to_raw='./structures/', n=None, prefix='full'):
    '''Generate raw unprocessed torch file to generate pyg datasets with fewer
        candidates.
    '''
    # Does this require a different dataset directory? Can try, just back up
    # structures.pt file.
    if n is None:
        t = None
    else:
        t = int(n/5)
    path_to_output = './datasets/{}_test/raw/'.format(prefix)
    test_indices = []
    test_structures = []
    idx = 0
    for path in tqdm(glob(path_to_raw + '/test/*')[:t], desc='Reading Structures'):
        name = path.rsplit('/', 1)[1].split('.')[0]
        test_structures.append(read_ply(path))
        test_indices.append((idx, name))
        idx += 1
    print('Saving test structures to file as pytorch object ...')
    torch.save(test_structures, path_to_output+'{}_structures.pt'.format(prefix))
    torch.save(test_indices, path_to_output+'{}_indices.pt'.format(prefix))
    path_to_output = './datasets/{}_train/raw/'.format(prefix)
    train_indices = []
    train_structures = []
    idx = 0
    for path in tqdm(glob(path_to_raw + '/train/*')[:n], desc='Reading Structures'):
        name = path.rsplit('/', 1)[1].split('.')[0]
        train_structures.append(read_ply(path))
        train_indices.append((idx, name))
        idx += 1
    print('Saving train structures to file as pytorch object ...')
    torch.save(train_structures, path_to_output+'{}_structures.pt'.format(prefix))
    torch.save(train_indices, path_to_output+'{}_indices.pt'.format(prefix))
    print('Done.')


def generate_numpy_from_structures(prefix='full'):
    train_structures = Structures(root='./datasets/{}_train'.format(prefix))
    train_indices = torch.load('./datasets/{}/raw/{}_indices.pt'.format(prefix))
    collection = []
    for data in tqdm(train_structures, desc='Converting train structures -> numpy'):
        cat = torch.cat((data.pos, data.norm, data.x, data.shape_index, data.y), dim=1).numpy()
        collection.append(cat)
    train_array = np.asarray(collection)

    test_structures = Structures(root='./datasets/{}_test'.format(prefix))
    test_indices = torch.load('./datasets/{}/raw/{}_indices.pt'.format(prefix))
    collection = []
    for data in tqdm(test_structures, desc='Converting test structures -> numpy'):
        cat = torch.cat((data.pos, data.norm, data.x, data.shape_index, data.y), dim=1).numpy()
        collection.append(cat)
    test_array = np.asarray(collection)


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
            s = slices[key][-1] + item[key].size(item.__cat_dim__(key, item[key]))
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


def read_ply(path, learn_iface=True):
    '''
        read_ply from pytorch_geometric does not capture the properties in ply
        file. This function adds to pyg's read_ply function by capturing extra
        properties: charge, hbond, hphob, and iface.

        # Update! Shape data should now be included as a transform.
    '''
    if learn_iface is False:
        raise NotImplementedError

    with open(path, 'rb') as f:
        data = PlyData.read(f)

    pos = ([torch.tensor(data['vertex'][axis]) for axis in ['x', 'y', 'z']])
    pos = torch.stack(pos, dim=-1)

    norm = ([torch.tensor(data['vertex'][axis]) for axis in ['nx', 'ny', 'nz']])
    norm = torch.stack(norm, dim=-1)

    x = ([torch.tensor(data['vertex'][axis]) for axis in ['charge', 'hbond', 'hphob']])
    x = torch.stack(x, dim=-1)
    y = None

    face = None
    if 'face' in data:
        faces = data['face']['vertex_indices']
        faces = [torch.tensor(fa, dtype=torch.long) for fa in faces]
        face = torch.stack(faces, dim=-1)

    data = Data(x=x, pos=pos, face=face, norm=norm, y=y)

    return data


class MiniStructures(InMemoryDataset):
    def __init__(self, root='./datasets/mini_pos/', transform=None, pre_transform=None):
        super(MiniStructures, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property  # What does the property decorator do?
    def raw_file_names(self):
        return ['mini_structures.pt']

    @property
    def processed_file_names(self):
        return ['mini_structures.pt']

    def download(self):
        pass

    def process(self):
        data_list = torch.load(self.raw_paths[0])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in data_list]

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])


class Structures(InMemoryDataset):
    def __init__(self, root='./datasets/{}/'.format(p.dataset), pre_transform=None, transform=None):
        self.prefix = p.dataset
        super(Structures, self).__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['{}structures.pt'.format(self.prefix + '_')]

    @property
    def processed_file_names(self):
        return ['{}structures.pt'.format(self.prefix + '_')]

    def download(self):
        pass

    def process(self):
        from utils import has_nan
        data_list = torch.load(self.raw_paths[0])

        if self.pre_filter is not None:
            data_list = [data for data in data_list if self.pre_filter(data)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(data) for data in tqdm(data_list)]

        try:
            if data_list[0].shape_index is not None:
                _, idx = has_nan(data_list)
                data_list = [data_list[i] for i in range(0, len(data_list)) if i not in idx]
        except AttributeError:
            pass

        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])
