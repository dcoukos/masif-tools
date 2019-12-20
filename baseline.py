import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from models import Basic_Net
from torch_geometric.transforms import FaceToEdge
from dataset import Structures, MiniStructures
from torch.utils.tensorboard import SummaryWriter, FileWriter
from sklearn.metrics import roc_curve, roc_auc_score
from utils import perf_measure, stats
'''
baseline.py implements a baseline model. Experiment using pytorch-geometric
    and FeaStNet.
'''
# Current goal: write metrics to tensorboard
# figure out how to observe metrics.

# TODO: send all tensors to device!
writer = SummaryWriter()
# Not able to add graph to writer.

samples = 50  # Doesn't currently do anything.
epochs = 300
batch_size = 200
validation_split = .2
shuffle_dataset = True
random_seed = 42
dropout = True
learning_rate = .1

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = MiniStructures(pre_transform=FaceToEdge())
samples = len(dataset)
if shuffle_dataset:
    dataset = dataset.shuffle()
n_features = dataset.get(0).x.shape[1]

model = Basic_Net(n_features, dropout=dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-4)

cutoff = int(np.floor(samples*(1-validation_split)))
train_dataset = dataset[:cutoff]
test_dataset = dataset[cutoff:]


train_loader = DataLoader(train_dataset, shuffle=True, batch_size=batch_size)
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

writer.add_graph(model, input_to_model=(x, edge_index, labels, torch.Tensor()))

for epoch in range(1, epochs+1):
    # rotate the structures between epochs
    model.train()
    # dataset_ = [converter(structure) for structure in dataset]
    first_batch_labels = torch.Tensor()
    pred = torch.Tensor()
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

    print("---- Round {}: loss={:.4f} ".format(epoch, loss))

    (train_TP, train_FP, train_TN, train_FN) = perf_measure(pred, first_batch_labels)
    print("Performance measures: {} {} {} {}".format(train_TP, train_FP, train_TN, train_FN))
    # print(stats(last_batch_labels, pred))

    model.eval()
    x, edge_index = test_data.x.to(device), test_data.edge_index.to(device)
    labels = test_data.y.to(device)
    _, out = model(x, edge_index, labels)
    pred = out.detach().round().to(device)

    (test_TP, test_FP, test_TN, test_FN) = perf_measure(pred, test_labels)

    #  --------------  REPORTING ------------------------------------

    writer.add_scalars('True positive rate', {'train': train_TP,
                                              'test': test_TP}, epoch)
    writer.add_scalars('False positive rate', {'train': train_FP,
                                               'test': test_FP}, epoch)
    writer.add_scalars('True negative rate', {'train': train_TN,
                                              'test': test_TN}, epoch)
    writer.add_scalars('False negative rate', {'train': train_FN,
                                               'test': test_FN}, epoch)
    # writer.add_scalars('Loss', {'train': })

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

'''
    correct = float(torch.tensor(pred.numpy().round()).eq(data.y).sum().item())
    incorrect = len(pred) - correct

    acc = correct / data.batch.size()[0]
    roc_auc = roc_auc_score(data.y, pred)
    fpr, tpr, thresholds = roc_curve(data.y, pred)


plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr, tpr)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.show()
print('Accuracy: {:.4f}'.format(acc))
print('ROC AUC: {:.3f}'.format(roc_auc))
'''

writer.close()
