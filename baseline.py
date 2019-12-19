import torch
import numpy as np
import matplotlib.pyplot as plt
from torch_geometric.data import DataLoader
from models import Basic_Net
from torch_geometric.transforms import FaceToEdge
from dataset import Structures
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_curve, roc_auc_score
'''
baseline.py implements a baseline model. Experiment using pytorch-geometric
    and FeaStNet.
'''
# Below not ready
writer = SummaryWriter()
# Not able to add graph to writer.

samples = 50  # Doesn't currently do anything.
epochs = 3
batch_size = 5
validation_split = .2
shuffle_dataset = True
random_seed = 42
dropout = False

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
dataset = Structures(root='./datasets/', pre_transform=FaceToEdge())

if shuffle_dataset:
    dataset = dataset.shuffle()
n_features = dataset.get(0).x.shape[1]

model = Basic_Net(n_features, dropout=dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=1e-4)

cutoff = int(np.floor(len(dataset)*(1-validation_split)))
train_dataset = dataset[:cutoff]
test_dataset = dataset[cutoff:]


train_loader = DataLoader(train_dataset, batch_size=batch_size)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

# Notes on training:
# The size of the input matrix = [n_nodes, n_x(explicit features)]
# rotate structures at each epoch
# converter = rotations

for epoch in range(epochs):
    # rotate the structures between epochs
    model.train()
    # dataset_ = [converter(structure) for structure in dataset]
    for data in train_loader:
        optimizer.zero_grad()
        loss, _ = model(data)
        loss.backward()
        optimizer.step()
    print("---- Round {}: loss={:.4f} ".format(epoch+1, loss))

model.eval()

# TODO: Implement testing metric.
# with torch.no_grad():
data = next(iter(test_loader))
_, out = model(data)
# pred = torch.tensor(out.detach().numpy().round())
pred = out.detach()

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
