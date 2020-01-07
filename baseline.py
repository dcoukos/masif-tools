import torch
import numpy as np
from decimal import Decimal
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from models import BasicNet, FeaStNet
from torch_geometric.transforms import FaceToEdge
from torch_geometric.utils import precision, recall, f1_score
from dataset import Structures, MiniStructures
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, roc_auc_score
from utils import perf_measure, stats
from glob import glob
import datetime
'''
baseline.py implements a baseline model. Experiment using pytorch-geometric
    and FeaStNet.
'''


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

samples = 50  # Doesn't currently do anything.
if str(device) == 'cuda':
    epochs = 300
else:
    epochs = 10
batch_size = 20
validation_split = .2
shuffle_dataset = False
random_seed = 42
dropout = False  # too much dropout?
learning_rate = .001
lr_decay = 0.99
weight_decay = 1e-4

dataset = MiniStructures(root='./datasets/mini_pos/', pre_transform=FaceToEdge())
# Add momentum? After a couple epochs, gradients locked in at 0.
samples = len(dataset)
if shuffle_dataset:
    dataset = dataset.shuffle()
n_features = dataset.get(0).x.shape[1]


model = BasicNet(n_features, dropout=False).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

writer = SummaryWriter(comment='model:{}_lr:{}_dr:{}_sh:{}'.format(str(type(model)).split('.')[1].split("\'")[0],
                                                                   learning_rate,
                                                                   dropout,
                                                                   shuffle_dataset))

cutoff = int(np.floor(samples*(1-validation_split)))
train_dataset = dataset[:cutoff]
test_dataset = dataset[cutoff:]


train_loader = DataLoader(train_dataset, shuffle=shuffle_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=len(test_dataset))


# Notes on training:
# The size of the input matrix = [n_nodes, n_x(explicit features)]
# rotate structures at each epoch
# converter = rotations

# to avoid continuous reloading
test_data = next(iter(test_loader))
test_labels = test_data.y.to(device)

# Adding graph to Tensorboard
datapoint = dataset.get(0)
x, edge_index = datapoint.x.to(device), datapoint.edge_index.to(device)
labels = datapoint.y.to(device)

# writer.add_graph(model, input_to_model=(x, edge_index, labels))

# previous loss stored for adaptive learning rate.
loss = 1
prev_loss = 1

# FOR DEVELOPMENT:
epochs = 1

for epoch in range(1, epochs+1):
    # rotate the structures between epochs
    model.train()
    # dataset_ = [converter(structure) for structure in dataset]
    first_batch_labels = torch.Tensor()
    pred = torch.Tensor()
    if prev_loss < loss:  # adaptive learning rate.
        for g in optimizer.param_groups:
            learning_rate = learning_rate*lr_decay
            g['lr'] = learning_rate
    prev_loss = loss
    for batch_n, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        labels = data.y.to(device)
        loss, out = model(x, edge_index, labels)
        loss.backward()
        optimizer.step()
        if batch_n == 0:
            first_batch_labels = data.y.clone().detach().to(device)
            pred = out.clone().detach().round().to(device)

    print("---- Round {}: loss={:.4f} lr:{:.4f}".format(epoch, loss, Decimal(learning_rate)))

    #  --------------  REPORTING ------------------------------------

    (train_TP, train_FP, train_TN, train_FN) = perf_measure(pred, first_batch_labels)
    print("Performance measures: {} {} {} {}".format(train_TP, train_FP, train_TN, train_FN))
    # print(stats(last_batch_labels, pred))
    train_precision = precision(pred, first_batch_labels, 2)[1].item()
    train_recall = recall(pred, first_batch_labels, 2)[1].item()
    train_f1 = f1_score(pred, first_batch_labels, 2)[1].item()

    model.eval()
    x, edge_index = test_data.x.to(device), test_data.edge_index.to(device)
    labels = test_data.y.to(device)
    _, out = model(x, edge_index, labels)
    pred = out.detach().round().to(device)

    (test_TP, test_FP, test_TN, test_FN) = perf_measure(pred, test_labels)
    test_precision = precision(pred, test_labels, 2)[1].item()
    test_recall = recall(pred, test_labels, 2)[1].item()
    test_f1 = f1_score(pred, test_labels, 2)[1].item()

    writer.add_scalars('True positive rate', {'train': train_TP,
                                              'test': test_TP}, epoch)
    writer.add_scalars('False positive rate', {'train': train_FP,
                                               'test': test_FP}, epoch)
    writer.add_scalars('True negative rate', {'train': train_TN,
                                              'test': test_TN}, epoch)
    writer.add_scalars('False negative rate', {'train': train_FN,
                                               'test': test_FN}, epoch)
    writer.add_scalars('Recall', {'train': train_recall,
                                  'test': test_recall}, epoch)
    writer.add_scalars('Precision', {'train': train_precision,
                                     'test': test_precision}, epoch)
    writer.add_scalars('F1_score', {'train': train_f1,
                                    'test': test_f1}, epoch)

    writer.add_histogram('Layer 1 weights', model.conv1.weight, epoch+1)
    writer.add_histogram('Layer 1 bias', model.conv1.bias, epoch+1)
    writer.add_histogram('Layer 1 weight gradients', model.conv1.weight.grad, epoch+1)

    writer.add_histogram('Layer 2 weights', model.conv2.weight, epoch+1)
    writer.add_histogram('Layer 2 bias', model.conv2.bias, epoch+1)
    writer.add_histogram('Layer 2 weight gradients', model.conv2.weight.grad, epoch+1)

    writer.add_histogram('Layer 3 weights', model.conv3.weight, epoch+1)
    writer.add_histogram('Layer 3 bias', model.conv3.bias, epoch+1)
    writer.add_histogram('Layer 3 weight gradients', model.conv3.weight.grad, epoch+1)

    writer.add_histogram('Layer 4 weights', model.lin1.weight, epoch+1)
    writer.add_histogram('Layer 4 bias', model.lin1.bias, epoch+1)
    writer.add_histogram('Layer 4 weight gradients', model.lin1.weight.grad, epoch+1)

    writer.add_histogram('Layer 5 weights', model.lin2.weight, epoch+1)
    writer.add_histogram('Layer 5 bias', model.lin2.bias, epoch+1)
    writer.add_histogram('Layer 5 weight gradients', model.lin2.weight.grad, epoch+1)

writer.close()

now = datetime.datetime.now().strftime('%y%m%d%H%M')
torch.save(model.state_dict(), 'models/{}_{}.pt'.format(str(model).split('(')[0], now))
