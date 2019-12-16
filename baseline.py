import torch
import glob
import numpy as np
import torch.nn.functional as F
from torch_geometric.data import DataLoader
from models import Basic_Net
from dataset import read_ply
from random import shuffle
from torch_geometric.transforms.face_to_edge import FaceToEdge

'''
baseline.py implements a baseline model. Experiment using pytorch-geometric
    and FeaStNet. Next implementation should use SpiralNet++, given data
    availability.

Todo:
    Goal: develop model for predicting interface in ply files.

    Step 1: Basline Implementation

    [x] Batch import the dataset
    [x] Figure out train/validation split
    [x] Create the network --> Current
    [x] Write a training step
    [] Implement Weight Bias initliazations.
    [x] Train the model
    [] Determine whether I should mask molecules or nodes.

    For next iteration:
    - Implement network from Feastnet.
    - Create a dataset, in-memory datset, and mini-dataset.

    Then:
    - Implement SpiralNet++ network.
    - Introduce custom pooling algorithm which does avg on the charge & max on
        the surface.
    - Introduce multi-scale architecture
    - adaptive learning rate.
    - evaluate optimizers other than Adam?
    - Data parallelism

'''
# Handle device.
epochs = 20
batch_size = 4
validation_split = .2
shuffle_dataset = False
random_seed = 42
n_features = 3  # Probably a smarter way to do this?

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Have to define the parameters. Preprocessing on the loaded data?
# https://pytorch.org/docs/master/data.html#data-loading-order-and-sampler

# have to define the number of features
model = Basic_Net(n_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

dataset = [read_ply(path, True) for path in
           glob.glob('../masif_pdbs_and_ply/benchmark_surfaces/*')[:20]]
converter = FaceToEdge()

# TR-VAL SPLIT V.01  cleaner?
dataset_ = [converter(structure) for structure in dataset]  # dataset
shuffle(dataset_)
cutoff = int(np.floor(len(dataset_)*(1-validation_split)))
train_dataset = dataset_[:cutoff]
test_dataset = dataset_[cutoff:]

len(train_dataset)
len(test_dataset)

train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=batch_size)


losses = []

# Notes on training:
    # The size of the input matrix = [n_nodes, n_x(explicit features)]
    # ERROR: AttributeError: 'NoneType' object has no attribute 'size'
    #           edge_index is NoneType
    # Something about converting graph from face something something.

for epoch in range(epochs):
    model.train()
    for data in train_loader:
        optimizer.zero_grad()
        loss = model(data)
        loss.backward()
        optimizer.step()
        losses.append(loss)
    print("---- Round {}: loss={:.4f} ".format(epoch+1, loss))

model.eval()


# TODO: Write testing metric.
data = next(iter(test_loader))

data
_, pred = model(data).max(dim=-1)
correct = float(pred.eq(data.y).sum().item())
acc = correct / data.batch.size()[0]
print('Accuracy: {:.4f}'.format(acc))
