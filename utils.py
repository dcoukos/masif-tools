import torch
import os
import params as p
from glob import glob
from torch_geometric.transforms import Compose, FaceToEdge, TwoHop, Center
from transforms import *
import datetime
import pathlib
from tqdm import tqdm
import pandas as pd


def generate_index_list(path_to_index_file):
    f = None
    if 'test' in path_to_index_file:
        f = list(open('./lists/masif_site_testing.txt', 'r'))
    else:
        f = list(open('./lists/masif_site_training.txt', 'r'))
    indices = pd.DataFrame(torch.load(path_to_index_file), columns=['index', 'name'])
    names = []
    saved_indices = []
    not_in_dataset = []
    for line in f:
        names.append(line.split('\n')[0])
    for name in names:
        try:
            i, = indices[indices.name == name].index
        except ValueError:
            not_in_dataset.append(name)
        saved_indices.append(i)
    return saved_indices, not_in_dataset


def apply_pretransforms(pre_transforms=None):
    from dataset import Structures
    # Structures should check already whether these pre_transforms have been computed
    if pre_transforms is None:
        trainset = Structures(root='./datasets/{}_train/'.format(p.dataset),
                                 pre_transform=Compose((FaceAttributes(),
                                                        NodeCurvature(), FaceToEdge(), TwoHop())))
        testset = Structures(root='./datasets/{}_test/'.format(p.dataset),
                                pre_transform=Compose((FaceAttributes(),
                                                       NodeCurvature(), FaceToEdge(), TwoHop())))
    else:
        trainset = Structures(root='./datasets/{}_train/'.format(p.dataset),
                                 pre_transform=pre_transforms)
        testset = Structures(root='./datasets/{}_test/'.format(p.dataset),
                                pre_transform=pre_transforms)
    return trainset, testset


def perf_measure(pred, labels):
    '''Calculates and returns performance metrics from two tensors when doing binary
        classification'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = pred == labels
    B = pred == torch.ones(size=pred.shape).to(device)

    TP = (A * B).sum().item()
    FP = (~A * B).sum().item()
    TN = (A * ~B).sum().item()
    FN = (~A * ~B).sum().item()

    return (TP, FP, TN, FN)


def stats(pred, labels):
    from sklearn.metrics import classification_report
    return classification_report(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())


def split_datasets():
    '''Need to call from inside masif-tools'''
    wd = pathlib.Path().absolute()
    #os.mkdir('{}/structures_copy/test'.format(wd))
    #os.mkdir('{}/structures_copy/train'.format(wd))
    test_paths = []
    missing_test_paths = []
    missing_train_paths = []

    with open('./lists/masif_site_testing.txt', 'r') as f:
        for line in f:
            line = line.split('\n')[0]
            test_paths.append(line)
    for name in tqdm(test_paths, desc='Moving test files'):
        if len(name.split('_')) == 3:
            a, b, c = name.split('_')
            name = a + '_' + b
            structure2 = a + '_' + c
            test_paths.append(structure2)
        new_path = os.path.expanduser('/Volumes/Storage/LPDI/masif-tools/structures_copy/test/') + name + \
            '.ply'
        path = os.path.expanduser('/Volumes/Storage/LPDI/masif-tools/structures_copy/') + name + \
            '.ply'
        try:
            os.replace(path, new_path)
        except:
            missing_test_paths.append(name)
    print('{} test paths missing'.format(len(missing_test_paths)))

    train_paths = []
    with open('./lists/masif_site_training.txt', 'r') as f:
        for line in f:
            line = line.split('\n')[0]
            train_paths.append(line)
    for name in tqdm(train_paths, desc='Moving train files'):
        if len(name.split('_')) == 3:
            a, b, c = name.split('_')
            name = a + '_' + b
            structure2 = a + '_' + c
            train_paths.append(structure2)
        new_path = os.path.expanduser('/Volumes/Storage/LPDI/masif-tools/structures_copy/train/') + name + \
            '.ply'
        path = os.path.expanduser('/Volumes/Storage/LPDI/masif-tools/structures_copy/') + name + \
            '.ply'
        try:
            os.replace(path, new_path)
        except FileNotFoundError:
            missing_train_paths.append(name)

    print('{} paths missing'.format(len(missing_train_paths)))
    return {'missing_test_paths': missing_test_paths,
            'missing_train_paths': missing_train_paths}
    # rest in command line
    # Move remaining structures to new directory called train.


def generate_weights(labels):
    '''
        Takes a label tensor, and generates scoring weights for binary_cross_entropy
    '''
    if p.interface_weight is None:
        example = MiniStructures()  # This must have made it slow.
        n_nodes = float(len(example.data.y))
        n_pos_nodes = example.data.y.sum().item()
        n_neg_nodes = n_nodes - n_pos_nodes
        ratio_neg = n_neg_nodes/n_nodes
        ratio_pos = n_pos_nodes/n_nodes
        weight_neg = ratio_pos
        weight_pos = ratio_neg
    else:
        weight_neg = 1 - p.interface_weight
        weight_pos = p.interface_weight - weight_neg

    return (labels.clone().detach()*weight_pos + weight_neg)


def generate_example_surfaces(model_type, model_path, n_examples=5, use_structural_data=False):
    '''
        Save graph vertices in ply file format. Loads a model from path and runs n_example
        structures through the model, and saves the graph vertices with the predicted surface
        interface labels.
    '''
    converter = FaceToEdge()

    paths = glob('./structures/test/*')[:n_examples]
    names = [path.split('/')[-1]for path in paths]
    structures = [read_ply(path) for path in paths]

    faces = [structure.face for structure in structures]
    structures = [converter(structure) for structure in structures]

    device = torch.device('cpu')
    structures[0].x.shape[1]
    if p.heads is not None:
        model = model_type(structures[0].x.shape[1], heads=p.heads)
    else:
        model = model_type(structures[0].x.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    predictions = []
    for data in structures:
        out = model(data)
        predictions.append(out)

    # ---- Make directory ---
    dir = model_path.split('models/', 1)[1]
    wd = pathlib.Path().absolute()
    full_path = '{}/surfaces/{}'.format(wd, dir)
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    for n, structure in enumerate(structures):
        rounded = predictions[n].round()
        save_ply(
            filename='./surfaces/{}/{}'.format(dir, names[n]),
            vertices=structure.pos.detach().numpy(),
            normals=structure.norm.detach().numpy(),
            faces=faces[n].t().detach().numpy(),
            charges=structure.x[:, 0].reshape(-1, 1).detach().numpy(),
            hbond=structure.x[:, 1].reshape(-1, 1).detach().numpy(),
            hphob=structure.x[:, 2].reshape(-1, 1).detach().numpy(),
            iface=predictions[n].detach().numpy()
        )

        save_ply(
            filename='./surfaces/{}/r_{}'.format(dir, names[n]),
            vertices=structure.pos.detach().numpy(),
            normals=structure.norm.detach().numpy(),
            faces=faces[n].t().detach().numpy(),
            charges=structure.x[:, 0].reshape(-1, 1).detach().numpy(),
            hbond=structure.x[:, 1].reshape(-1, 1).detach().numpy(),
            hphob=structure.x[:, 2].reshape(-1, 1).detach().numpy(),
            iface=rounded.detach().numpy()
        )


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


def make_model_directory(model_folder='models'):
    ''' make directory to save models.
    '''
    now = datetime.datetime.now().strftime('%b%d_%H:%M')
    modelpath = '{}/{}_{}'.format(model_folder, now, str(p.version).split('(')[0])
    surfacepath = 'surfaces/{}_{}'.format(now, str(p.version).split('(')[0])
    wd = pathlib.Path().absolute()
    os.mkdir('{}/{}'.format(wd, modelpath))
    os.mkdir('{}/{}'.format(wd, surfacepath))
    return modelpath


def generate_surface(model_type, model_path, pdb_code, use_structural_data=False):
    '''
        Save the surface prediction for a particular structure.
    '''
    from dataset import read_ply
    '''
    converter = Compose((Center(), FaceAttributes(),
                        NodeCurvature(), FaceToEdge(),
                        TwoHop(), AddShapeIndex()))
    '''
    converter = Compose((Center(), FaceAttributes(),
                        NodeCurvature(), FaceToEdge(),
                        TwoHop(), AddShapeIndex()))
    path = glob('./structures/test/{}.ply'.format(pdb_code))[0]
    name = path.split('/')[-1]
    structure = read_ply(path)

    face = structure.face
    structure = converter(structure)

    device = torch.device('cpu')
    structure.x.shape[1]
    if p.heads is not None:
        model = model_type(structure.x.shape[1], heads=p.heads)
    else:
        model = model_type(structure.x.shape[1])
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    prediction = model(structure)
    rounded = prediction.round()

    # ---- Make directory ---
    dir = model_path.split('models/', 1)[1].split('.')[0]
    full_path = os.path.expanduser('~/Desktop/Drawer/LPDI/masif-tools/surfaces/' + dir)
    folder = full_path.rsplit('/', 1)[0]
    if not os.path.exists(folder):
        os.mkdir(folder)
    if not os.path.exists(full_path):
        os.mkdir(full_path)

    save_ply(
        filename='./surfaces/{}/{}'.format(dir, name),
        vertices=structure.pos.detach().numpy(),
        normals=structure.norm.detach().numpy(),
        faces=face.t().detach().numpy(),
        charges=structure.x[:, 0].reshape(-1, 1).detach().numpy(),
        hbond=structure.x[:, 1].reshape(-1, 1).detach().numpy(),
        hphob=structure.x[:, 2].reshape(-1, 1).detach().numpy(),
        iface=prediction.detach().numpy()
    )

    save_ply(
        filename='./surfaces/{}/r_{}'.format(dir, name),
        vertices=structure.pos.detach().numpy(),
        normals=structure.norm.detach().numpy(),
        faces=face.t().detach().numpy(),
        charges=structure.x[:, 0].reshape(-1, 1).detach().numpy(),
        hbond=structure.x[:, 1].reshape(-1, 1).detach().numpy(),
        hphob=structure.x[:, 2].reshape(-1, 1).detach().numpy(),
        iface=rounded.detach().numpy()
    )


def has_nan(dataset):
    assert dataset[0].shape_index is not None
    max_ = 0
    idx = []
    for i, data in tqdm(enumerate(dataset)):
        has_nan = max(torch.isnan(data.shape_index))
        if has_nan:
            idx.append(i)
        max_ = max(has_nan, max_)
    return max_, idx
