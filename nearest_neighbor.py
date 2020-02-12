import torch
import Bio.SeqUtils
import torch_geometric
from glob import glob
import Bio.PDB as pdb
from biopandas.pdb import PandasPdb
from dataset import read_ply
from tqdm import tqdm
from sklearn import preprocessing
import numpy as np

# Define the LabelEncoder
ppdb = PandasPdb()
le = preprocessing.LabelEncoder()
paths = glob('../masif_site_masif_search_pdbs_and_ply_files/01-benchmark_pdbs/*')

cpu = torch.device('cpu')
gpu = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = paths[2]
mol_name = path.rsplit('/', 1)[1].split('.')[0]
structure = None
te_structure = None
le_defined = False

iter_paths = iter(paths)

residue_names = np.array(['LYS', 'GLU', 'ASP', 'SER', 'PHE', 'CYS', 'VAL', 'ILE', 'MET',
       'HIS', 'GLY', 'LEU', 'TYR', 'THR', 'PRO', 'ARG', 'TRP', 'ALA',
       'GLN', 'ASN', 'SEC'], dtype=object)

le.fit(residue_names)

# load from file... map with sklearn. Save to file as pt. Save all structures.
train_structures = []
test_structures = []

for path in tqdm(paths):
    try:
        train_bool, structure = get_neighbors(path, device)
    except RuntimeError as e:
        if 'memory' in str(e):
            train_bool, structure = get_neighbors(path, torch.device('cpu'))
            print('Large structure. Running this structure on cpu.')
        else:
            raise e

    if train_bool is True:
        train_structures.append(structure)
    else:
        test_structures.append(structure)

torch.save(train_structures, './datasets/res_train/raw/res_structures.pt')
torch.save(test_structures, './datasets/res_test/raw/res_structures.pt')


def get_neighbors(path, device):
    ppdb.read_pdb(path=path)
    # Load through read_ply function.

    mol_name = path.rsplit('/', 1)[1].split('.')[0]
    train = True
    try:
        structure = read_ply('./structures/train/{}.ply'.format(mol_name))

    except FileNotFoundError:
        structure = read_ply('./structures/test/{}.ply'.format(mol_name))
        train = False

    nodes = structure.pos.to(gpu).float()
    n_nodes = nodes.shape[0]

    pos = ['x_coord', 'y_coord', 'z_coord']

    atoms = torch.tensor(ppdb.df['ATOM'][pos].values).to(gpu).float()

    atom_shape = atoms.shape
    atoms = atoms.view(-1, 1, 3).expand(-1, n_nodes, 3)

    closest_atom = (atoms-nodes).norm(dim=2).argmin(dim=0)

    structure_residues = ppdb.df['ATOM'][['atom_number', 'residue_name']]
    n_atoms = structure_residues.shape[0]
    idx_translation = torch.LongTensor(le.transform(structure_residues.residue_name) + 1).to(gpu)

    node_idx = torch.tensor(range(0, n_nodes)).to(gpu)
    node_idx = torch.stack((node_idx, closest_atom)).t()

    closest_atom_sparse = torch.sparse.LongTensor(node_idx.t(),
                                                  torch.ones(n_nodes, dtype=torch.long).to(gpu),
                                                  torch.Size([n_nodes, n_atoms])).to(gpu)
    amino_acids = (closest_atom_sparse.to_dense() *
                   idx_translation.view(-1, 1).t()).to_sparse().values().to(cpu)
    structure.residues = amino_acids

    if train is True:
        return train, structure
    else:
        return train, structure
