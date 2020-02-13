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

# Define the LabelEncoder

def get_neighbors(path, device, label_encoder):
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


ppdb = PandasPdb()
paths = glob('../masif_site_masif_search_pdbs_and_ply_files/01-benchmark_pdbs/*')

cpu = torch.device('cpu')
gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

res_encoder = {'LYS': 1, 'GLU': 2, 'ASP': 3, 'SER': 4, 'PHE': 5,
                'CYS': 6, 'VAL': 7, 'ILE': 8, 'MET': 9, 'HIS': 10,
                'GLY': 11, 'LEU': 12, 'TYR': 13, 'THR': 14, 'PRO': 15,
                'ARG': 16, 'TRP': 17, 'ALA': 18, 'GLN': 19, 'ASN': 20,
                'SEC': 21, 'UNK': 21, 'ASX': 21, 'GLX': 21, 'XLE': 21,
                'PYL': 21}

# load from file... map with sklearn. Save to file as pt. Save all structures.
train_structures = []
test_structures = []

for path in tqdm(paths):
    try:
        train_bool, structure = get_neighbors(path, gpu, res_encoder)
    except RuntimeError as e:
        print('Large structure exhausted CUDA memory. Running this structure on cpu.\t', end='')
        tic = time.time()
        train_bool, structure = get_neighbors(path, cpu, res_encoder)
        toc = time.time()
        print('Time elapsed: {}'.format(toc-tic))
        logging.info(e, exc_info=True)

    if train_bool is True:
        train_structures.append(structure)
    else:
        test_structures.append(structure)

torch.save(train_structures, './datasets/res_train/raw/res_structures.pt')
torch.save(test_structures, './datasets/res_test/raw/res_structures.pt')
