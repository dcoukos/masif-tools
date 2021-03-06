import time
import logging
import torch
import Bio.SeqUtils
import torch_geometric
from glob import glob
import Bio.PDB as pdb
from biopandas.pdb import PandasPdb
from dataset import read_ply
from tqdm import tqdm
import numpy as np


def get_neighbors(path, device):
    res_encoder =  {'LYS': 1, 'GLU': 2, 'ASP': 3, 'SER': 4, 'PHE': 5,
                    'CYS': 6, 'VAL': 7, 'ILE': 8, 'MET': 9, 'HIS': 10,
                    'GLY': 11, 'LEU': 12, 'TYR': 13, 'THR': 14, 'PRO': 15,
                    'ARG': 16, 'TRP': 17, 'ALA': 18, 'GLN': 19, 'ASN': 20,
                    'SEC': 21, 'UNK': 21, 'ASX': 21, 'GLX': 21, 'XLE': 21,
                    'PYL': 21}

    ppdb = PandasPdb()
    ppdb.read_pdb(path=path)
    # Load through read_ply function.

    mol_name = path.rsplit('/', 1)[1].split('.')[0]
    train = True
    try:
        structure = read_ply('./structures/train/{}.ply'.format(mol_name))

    except FileNotFoundError:
        structure = read_ply('./structures/test/{}.ply'.format(mol_name))
        train = False
    nodes = structure.pos.to(device).float()
    n_nodes = nodes.shape[0]

    pos = ['x_coord', 'y_coord', 'z_coord']

    atoms = torch.tensor(ppdb.df['ATOM'][pos].values).to(device).float()

    atom_shape = atoms.shape
    atoms = atoms.view(-1, 1, 3).expand(-1, n_nodes, 3)

    closest_atom = (atoms-nodes).norm(dim=2).argmin(dim=0)

    structure_residues = ppdb.df['ATOM'][['atom_number', 'residue_name']]
    n_atoms = structure_residues.shape[0]
    idx_translation = torch.LongTensor(structure_residues.residue_name.
                                       replace(res_encoder)).to(device)

    node_idx = torch.tensor(range(0, n_nodes)).to(device)
    node_idx = torch.stack((node_idx, closest_atom)).t()

    closest_atom_sparse = torch.sparse.LongTensor(node_idx.t(),
                                                  torch.ones(n_nodes, dtype=torch.long).to(device),
                                                  torch.Size([n_nodes, n_atoms])).to(device)

    amino_acids = (closest_atom_sparse.to_dense() * idx_translation.view(-1, 1).t()).to_sparse().values().to(cpu)
    structure.residues = amino_acids

    if train is True:
        return train, structure
    else:
        return train, structure


paths = glob('../masif_site_masif_search_pdbs_and_ply_files/01-benchmark_pdbs/*')

cpu = torch.device('cpu')
gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# load from file... map with sklearn. Save to file as pt. Save all structures.
train_idx, test_idx = 0, 0

for path in tqdm(paths):
    try:
        train_bool, structure = get_neighbors(path, gpu)
    except RuntimeError as e:
        print(e)
        print("Rerunning on cpu")
        train_bool, structure = get_neighbors(path, cpu)

    if train_bool is True:
        torch.save(structure, './datasets/res_train/raw/res_structures_{}.pt'.format(train_idx))
        train_idx += 1
    else:
        torch.save(structure, './datasets/res_test/raw/res_structures_{}.pt'.format(test_idx))
        test_idx += 1

    try:
        torch.cuda.empty_cache()
    except RuntimeError:
        raise RuntimeError
