import torch
import numpy as np
from torch_geometric.data import DataLoader
from models import SixConvResidual, ThreeConvGlobal  #, MiniModel
from torch_geometric.transforms import FaceToEdge
from dataset import MiniStructures
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from utils import generate_weights
import datetime
import params as p
from statistics import mean
'''
baseline.py implements a baseline model. Experiment using pytorch-geometric
    and FeaStNet.
Focus on just 1 conv small conv layer first! Learn on 1 structure, then 10 structures.
Then increase model complexity to accomodate the increase in data.
Ignore test metrics for now.
'''

if p.suppress_warnings:
    import warnings
    warnings.filterwarnings("ignore")

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# reproducibility
torch.manual_seed(p.random_seed)
np.random.seed(p.random_seed)

learn_rate = p.learn_rate

if str(device) == 'cuda':
    epochs = p.epochs
else:
    epochs = 20

dataset = MiniStructures(root='./datasets/mini/', pre_transform=FaceToEdge())
samples = len(dataset)
if p.shuffle_dataset:
    dataset = dataset.shuffle()
n_features = dataset.get(0).x.shape[1]

model = SixConvResidual(n_features, heads=1, dropout=p.dropout).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=p.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=p.lr_decay,
                                                       patience=p.patience)

writer = SummaryWriter(comment='model:{}_lr:{}_lr_decay:{}'.format(
                       p.version,
                       learn_rate,
                       p.lr_decay))

cutoff = int(np.floor(samples*(1-p.validation_split)))
train_dataset = dataset[:cutoff]
test_dataset = dataset[cutoff:]


train_loader = DataLoader(train_dataset, shuffle=p.shuffle_dataset, batch_size=p.batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=p.test_batch_size)

for epoch in range(1, epochs+1):
    # rotate the structures between epochs

    model.train()
    first_batch_labels = torch.Tensor()
    pred = torch.Tensor()
    tr_weights = torch.Tensor()
    loss = []

    for batch_n, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        labels = data.y.to(device)
        weights = generate_weights(labels)
        tr_loss, out = model(x, edge_index, labels, weights)
        loss.append(tr_loss.detach().item())
        tr_loss.backward()
        optimizer.step()
        if batch_n == 0:
            tr_weights = weights
            first_batch_labels = data.y.clone().detach().to(device)
            pred = out.clone().detach().round().to(device)

    loss = mean(loss)
    print("---- Round {}: loss={:.4f} lr:{:.6f}"
          .format(epoch, loss, learn_rate))

    #  --------------  REPORTING ------------------------------------

    roc_auc = roc_auc_score(first_batch_labels.cpu(), pred.cpu(), sample_weight=tr_weights.cpu())

    model.eval()
    cum_pred = torch.Tensor().to(device)
    cum_labels = torch.Tensor().to(device)
    for batch_n, data in enumerate(test_loader):
        x, edge_index = data.x.to(device), data.edge_index.to(device)
        labels = data.y.to(device)
        te_weights = generate_weights(labels)
        te_loss, out = model(x, edge_index, labels, te_weights)
        pred = out.detach().round().to(device)
        cum_labels = torch.cat((cum_labels, labels.clone().detach()), dim=0)
        cum_pred = torch.cat((cum_pred, pred.clone().detach()), dim=0)

    te_weights = generate_weights(cum_labels)
    roc_auc_te = roc_auc_score(cum_labels.cpu(), cum_pred.cpu(), sample_weight=te_weights.cpu())

    writer.add_scalars('Loss', {'train': tr_loss,
                                'test': te_loss}, epoch)
    writer.add_scalars('ROC AUC', {'train': roc_auc,
                                   'test': roc_auc_te}, epoch)
    writer.add_scalar('learning rate', learn_rate, epoch)

writer.close()

now = datetime.datetime.now().strftime('%y%m%d%H%M')
path = 'models/{}_{}.pt'.format(str(model).split('(')[0], now)
with open(path, 'a+'):
    torch.save(model.state_dict(), path)
