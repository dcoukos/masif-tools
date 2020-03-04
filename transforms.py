import torch
from torch_geometric.nn.conv.ppf_conv import point_pair_features
import math
import numpy as np
import torch.sparse as tsp
from models import ThreeConvBlock
from glob import glob


class FaceAttributes(object):
    '''
    Add curvature attributes and weights to each face.

    Not tested on GPU.
    '''
    def __init__(self):
        print('Calculating Shape Indices')

    def __call__(self, data):
        assert data.face is not None
        assert data.pos is not None
        assert data.norm is not None

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')

        faces = data.pos[data.face].to(device)  # checked.
        norms = data.norm[data.face].to(device)  # checked.

        e0 = faces[2,:,:] - faces[1,:,:]  # checked
        e1 = faces[0,:,:] - faces[2,:,:]  # checked
        e2 = faces[1,:,:] - faces[0,:,:]  # checked
        # data.face_normals = e0.cross(e1)  # checked

        n0 = norms[0,:,:]  # checked
        n1 = norms[1,:,:]  # checked
        n2 = norms[2,:,:]  # checked

        # Gram Schmidt Method to find an orthonormal basis.
        u = e0  # checked
        # u = torch.div(u, u.norm(dim=1).view(-1, 1))
        # u is already normalized.

        v = e1 - ((e1*u).sum(-1)/(u*u).sum(-1)).view(-1, 1)*u  # checked dims, can check calc.
        v = torch.div(v, v.norm(dim=1).view(-1, 1))  # checked.

        a_0 = (e0*u).sum(-1)
        a_1 = (e1*u).sum(-1)
        a_2 = (e2*u).sum(-1)
        a_3 = (e0*v).sum(-1)
        a_4 = (e1*v).sum(-1)
        a_5 = (e2*v).sum(-1)

        A = torch.stack((torch.stack((a_0, a_1), dim=1),
                         torch.stack((a_2, a_3), dim=1),
                         torch.stack((a_4, a_5), dim=1)), dim=1)
        b_0 = ((n2 - n1)*u).sum(-1)
        b_1 = ((n0 - n2)*u).sum(-1)
        b_2 = ((n1 - n0)*u).sum(-1)

        b = torch.stack((b_0, b_1, b_2), dim=1).view(-1, 3, 1)
        Dn_u = torch.pinverse(torch.transpose(A, 1, 2)@A)@torch.transpose(A, 1, 2)@b

        b_0 = ((n2 - n1)*v).sum(-1)
        b_1 = ((n0 - n2)*v).sum(-1)
        b_2 = ((n1 - n0)*v).sum(-1)

        b = torch.stack((b_0, b_1, b_2), dim=1).view(-1, 3, 1)
        Dn_v = torch.pinverse(torch.transpose(A, 1, 2)@A)@torch.transpose(A, 1, 2)@b

        data.face_curvature = torch.cat((Dn_u, Dn_v), dim=1).squeeze()
        s = 0.5 * (e0.norm(dim=1) + e1.norm(dim=1) + e2.norm(dim=1))
        data.face_weight = torch.sqrt(s*(s-e0.norm(dim=1))*(s-e1.norm(dim=1))*(s-e2.norm(dim=1)))

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class NodeCurvature(object):
    '''
        Computes the shape index for each node.

        Not tested on GPU.
    '''
    def __init__(self, remove_face_data=True):
        self.remove = remove_face_data

    def __call__(self, data):
        assert data.face is not None
        assert data.face_curvature is not None
        assert data.face_weight is not None

        # Prepare the initial local coordinate system
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        # device = torch.device('cpu')
        norms = data.norm.to(device)
        positions = data.pos.to(device)
        faces = data.face.to(device)

        face_id = torch.tensor(list(range(len(faces.t())))).to(device)  # checked
        face0 = faces[0]  # checked
        face1 = faces[1]  # checked
        face2 = faces[2]  # checked

        weights = data.face_weight  # checked
        f_curv = data.face_curvature

        face0 = torch.stack((face_id, face0), dim=0).t()  # checked
        face1 = torch.stack((face_id, face1), dim=0).t()  # checked
        face2 = torch.stack((face_id, face2), dim=0).t()  # checked
        weights = data.face_weight*torch.ones(len(face0), dtype=torch.long).to(device)
        sparse_size = torch.Size((faces.shape[1], len(positions)))  # checked

        sparse_face0 = tsp.FloatTensor(torch.LongTensor(face0).t(), weights, sparse_size).to(device)  # .to_dense()  # checked
        sparse_face1 = tsp.FloatTensor(torch.LongTensor(face1).t(), weights, sparse_size).to(device) # .to_dense()  # checked
        sparse_face2 = tsp.FloatTensor(torch.LongTensor(face2).t(), weights, sparse_size).to(device)  # .to_dense()  # checked

        weighted_faces = sparse_face0 + sparse_face1 + sparse_face2  # checked
        weighted_faces = weighted_faces.coalesce()

        # checked On older pytorch have to cast to float
        weighted_faces = weighted_faces.t()
        node_curv = tsp.mm(weighted_faces, f_curv)
        sum_weights_per_node = tsp.sum(weighted_faces, dim=1).to_dense()  # checked
        node_curv = node_curv.t()/sum_weights_per_node  # checked
        node_curv = node_curv.t()
        eigs = []
        for i in node_curv:  # checked
            eig = torch.eig(i.reshape(2,2))
            principal_curvatures = eig.eigenvalues[:,0].sort(descending=True).values
            eigs.append(principal_curvatures)
        eigs = torch.stack(eigs, dim=0)
        s_s = eigs[:,0] + eigs[:,1]
        s_p = eigs[:,0] - eigs[:,1]
        s = s_s.div(s_p)
        pi = math.pi*torch.ones(len(positions)).to(device)
        s = (2/pi)*torch.atan(s)

        data.shape_index = s
        if self.remove:
            data.face_curvature = None
            data.face_weights = None
            data.face_normals = None
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class AddShapeIndex(object):
    def __call__(self, data):
        assert data.shape_index is not None
        s = data.shape_index
        x = data.x
        s = s.view(-1, 1)

        x = torch.cat((x, s), dim=1)
        data.x = x
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class RemovePositionalData(object):
    ''''''
    def __call__(self, data, shape_index=True):
        len_ = 4 if shape_index else 3
        x = data.x
        y = x.narrow(1, 0, len_).clone()
        data.x = y
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class RemoveXYZ(object):
    ''''''
    def __call__(self, data):
        x = data.x
        y = x.narrow(1, 3, 6).clone()
        data.x = y
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class AddPositionalData(object):
    ''''''
    def __call__(self, data):
        pos = data.pos
        norm = data.norm
        x = data.x
        n_features = x.shape[1]

        x = torch.cat((x, pos, norm), dim=1)  # Potential error here!!
        data.x = x
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class BlockModelApply(object):
    '''
        Runs data through model, which is saved per-block in various torch files.
        Returns treated data from pre-output layer.
    '''
    def __init__(self, model_parameters, saved_model_paths):
        super(BlockModelApply, self).__init__()
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.models = [ThreeConvBlock(*model_parameters) for path in saved_model_paths]
        self.prepare_models_(saved_model_paths)

    def prepare_models_(self, paths):
        for model, path in zip(self.models, paths):
            model = model.to(self.device)
            model.load_state_dict(torch.load(path, map_location=self.device))
            model.eval()

    def __call__(self, data):
        for model in self.models:
            data = data.to(self.device)
            _, inter = model(data)
            data.x += inter

        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class RemoveFeatures(object):
    def __init__(self, columns):
        super(RemoveFeatures, self).__init__()
        self.to_remove = columns

    def __call__(self, data):
        n_features = data.x.shape[1]
        idx = [i for i in range(0, n_features)]
        if isinstance(self.to_remove, int):
            idx.pop(self.to_remove)
        elif isinstance(self.to_remove, list):
            self.to_remove.sort(reverse=True)
            for i in self.to_remove:
                idx.pop(i)
        else:
            raise RuntimeError('"columns" attribute must be int or list of int.')
        idx = torch.tensor(idx)

        data.x = data.x[:, idx]
        return data


class AddMasifDescriptor(object):
    def __init__(self, remove_other_features):
        super(AddMasifDescriptor, self).__init__()
        self.clean = remove_other_features

    def __call__(self, data):
        assert data.name is not None

        pdb = data.name.split('_')[0]
        chain = data.name.split('_')[1]

        folder_list = glob('./all_feat/{}*'.format(pdb))
        try:
            assert len(folder_list) == 1
        except AssertionError:
            print(folder_list)
            return None
        try:
            folder = folder_list[0]
        except IndexError:
            print(data.name)
            return None
        _, chA, chB = folder.rsplit('/', 1)[1].split('_')
        descriptor = None
        if chain == chA:
            descriptor = torch.tensor(np.load('{}/p1_desc_straight.npy'.format(folder)))
        if chain == chB:
            descriptor = torch.tensor(np.load('{}/p2_desc_straight.npy'.format(folder)))
        try:
            assert data.x.shape[0] == descriptor.shape[0]
        except AttributeError:
            return None
        if self.clean:
            data.x = descriptor
        else:
            data.x = torch.cat((data.x, descriptor), dim=1)

        return data


class AddRandomFeature(object):
    def __init__(self):
        super(AddRandomFeature, self).__init__()

    def __call__(self, data):
        x = data.x
        rand = torch.zeros((x.shape[0], 1), dtype=torch.short).random_()
        data.x = torch.cat((x, rand), dim=1)
        return data
