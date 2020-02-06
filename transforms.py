import torch
from torch_geometric.nn.conv.ppf_conv import point_pair_features
import math
import numpy as np
from torch.sparse import LongTensor


class FaceAttributes(object):
    '''
    Add curvature attributes and weights to each face.
    '''
    def __init__(self):
        print('Calculating Shape Indices')


    def __call__(self, data):
        assert data.face is not None
        assert data.pos is not None
        assert data.norm is not None

        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')

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
        print((torch.transpose(A, 1, 2)@A).shape)
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
        Computes the shape index for each node
    '''
    def __init__(self, remove_face_data=True):
        self.remove = remove_face_data

    def __call__(self, data):
        assert data.face is not None
        assert data.face_curvature is not None
        assert data.face_weight is not None

        # Prepare the initial local coordinate system
        #device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        device = torch.device('cpu')
        norms = data.norm.to(device)
        positions = data.pos.to(device)
        faces = data.face.to(device)

        face_id = torch.tensor(list(range(len(faces.t()))))  # checked
        face0 = faces[0]  # checked
        face1 = faces[1]  # checked
        face2 = faces[2]  # checked

        face0 = torch.stack((face_id, face0), dim=0).t()  # checked
        face1 = torch.stack((face_id, face1), dim=0).t()  # checked
        face2 = torch.stack((face_id, face2), dim=0).t()  # checked
        ones = torch.ones(len(face0), dtype=torch.long)
        sparse_idx = torch.LongTensor(face0)  # checked
        sparse_size = torch.Size((faces.shape[1], len(positions)))  # checked

        sparse_face0 = LongTensor(torch.LongTensor(face0).t(), ones, sparse_size).to_dense()  # checked
        sparse_face1 = LongTensor(torch.LongTensor(face1).t(), ones, sparse_size).to_dense()  # checked
        sparse_face2 = LongTensor(torch.LongTensor(face2).t(), ones, sparse_size).to_dense()  # checked

        sparse_faces = sparse_face0 + sparse_face1 + sparse_face2  # checked

        weights = data.face_weight  # checked
        f_curv = data.face_curvature  # checked
        weights = weights.view(-1,1)*sparse_faces.to(torch.float)  # checked On older pytorch have to cast to float
        node_curv = f_curv.t()@weights  # checked
        sum_weights_per_node = weights.sum(0)  # checked
        node_curv = node_curv/sum_weights_per_node  # checked
        node_curv = node_curv.t()  # checked.
        eigs = []
        for i in node_curv:  # checked
            eig = torch.eig(i.reshape(2,2))
            principal_curvatures = eig.eigenvalues[:,0].sort(descending=True).values
            eigs.append(principal_curvatures)
        eigs = torch.stack(eigs, dim=0)
        s_s = eigs[:,0] + eigs[:,1]
        s_p = eigs[:,0] - eigs[:,1]
        s = s_s.div(s_p)
        pi = math.pi*torch.ones(len(positions))
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
    def __call__(self):
        assert data.shape_index is not None
        s = data.shape_index
        x = data.x

        x = torch.stack((x, s), dim=1)
        x = x.reshape(-1, 4)
        data.x = x
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)


class RemovePositionalData(object):
    ''''''
    def __call__(self, data):
        x = data.x
        y = x.narrow(1, 6, 3).clone()
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

        x = torch.stack((x, pos, norm), dim=1)
        x = x.reshape(-1, n_features+6)
        data.x = x
        return data

    def __repr__(self):
        return '{}()'.format(self.__class__.__name__)
