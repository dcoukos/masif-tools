import torch
from sklearn.metrics import classification_report
from dataset import MiniStructures, read_ply
from torch_geometric.transforms import FaceToEdge
from glob import glob


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
    return classification_report(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())


def generate_weights(labels):
    '''
        Takes a label tensor, and generates scoring weights for binary_cross_entropy
    '''
    example = MiniStructures()
    n_nodes = float(len(example.data.y))
    n_pos_nodes = example.data.y.sum().item()
    n_neg_nodes = n_nodes - n_pos_nodes
    ratio_neg = n_neg_nodes/n_nodes
    ratio_pos = n_pos_nodes/n_nodes

    return (labels.clone().detach()*ratio_neg + ratio_pos)


def generate_example_surfaces(model_type, path, n_examples=5):
    '''
        Save graph vertices in ply file format. Loads a model from path and runs n_example
        structures through the model, and saves the graph vertices with the predicted surface
        interface labels.
    '''
    converter = FaceToEdge()

    paths = glob('./structures/*')[:n_examples]
    names = [path.split('/')[-1]for path in paths]
    structures = [read_ply(path, use_shape_data=True) for path in paths]
    faces = torch.tensor([structure.face for structure in structures])
    structures = [converter(structure) for structure in structures]

    device = torch.device('cpu')
    model = model_type(structures[0].x.shape[1])
    model.load_state_dict(torch.load(path))
    model.eval()

    predictions = []
    for data in structures:
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        labels = data.y.to(device)
        loss, out_ = model(x, edge_index, labels)
        predictions.append(out_)

    for n, structure in enumerate(structures):
        save_ply(
            filename='./example_surfaces/{}'.format(names[n]),
            vertices=structure.pos.detach().numpy(),
            normals=structure.norm.detach().numpy(),
            faces=faces[n].t().detach().numpy(),
            charges=structure.x[:, 0].reshape(-1, 1).detach().numpy(),
            hbond=structure.x[:, 1].reshape(-1, 1).detach().numpy(),
            hphob=structure.x[:, 2].reshape(-1, 1).detach().numpy(),
            iface=predictions[n].detach().numpy()
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
