import torch
import Bio.SeqUtils
import torch_geometric
from glob import glob
import Bio.PDB as pdb
from biopandas.pdb import PandasPdb
from dataset import read_ply
from tqdm import tqdm
from sklearn import preprocessing

# Define the LabelEncoder
ppdb = PandasPdb()
le = preprocessing.LabelEncoder()
paths = glob('../masif_site_masif_search_pdbs_and_ply_files/01-benchmark_pdbs/*')

cpu = torch.device('cpu')
gpu = device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

path = paths[2]
mol_name = path.rsplit('/', 1)[1].split('.')[0]
structure = None
te_structure = None
le_defined = False

iter_paths = iter(paths)

# Fit Label Encoder!
while not le_defined:
    try:
        print('looping...')
        structure = None
        te_structure = None
        path = next(iter_paths)
        mol_name = path.rsplit('/', 1)[1].split('.')[0]

        structure = read_ply('./structures/train/{}.ply'.format(mol_name))
        ppdb.read_pdb(path=path)
        residue_names = ppdb.df['ATOM']['residue_name'].unique()
        if len(residue_names) == 20:
            print(True)
            le.fit(residue_names)
            le_defined = True
        else:
            print('Only {} AAs!'.format(len(residue_names)))
            continue

    except FileNotFoundError:
        te_structure = read_ply('./structures/test/{}.ply'.format(mol_name))
        print('In test set..')

# load from file... map with sklearn. Save to file as pt. Save all structures.
train_structures = []
test_structures = []

paths = paths[:100]

for path in tqdm(paths):
    ppdb.read_pdb(path=path)
    # Load through read_ply function.

    mol_name = path.rsplit('/', 1)[1].split('.')[0]
    train = True
    try:
        structure = read_ply('./structures/train/{}.ply'.format(mol_name))

    except FileNotFoundError:
        structure = read_ply('./structures/test/{}.ply'.format(mol_name))
        train = False

    nodes = structure.pos.to(gpu)
    n_nodes = nodes.shape[0]

    pos = ['x_coord', 'y_coord', 'z_coord']

    atoms = torch.tensor(ppdb.df['ATOM'][pos].values).to(gpu)

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
        train_structures.append(structure)
    else:
        test_structures.append(structure)

torch.save(train_structures, './datasets/res_train/raw/res_structures.pt')
torch.save(test_structures, './datasets/res_test/raw/res_structures.pt')
