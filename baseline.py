import torch
import numpy as np
from torch_geometric.data import DataLoader
from models import TwoConv
from torch_geometric.transforms import FaceToEdge
from torch_geometric.utils import precision, recall, f1_score
from dataset import MiniStructures
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from utils import generate_weights
import datetime
import params as p
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
lr = p.learn_rate

if str(device) == 'cuda':
    epochs = p.epochs
else:
    epochs = 20

dataset = MiniStructures(root='./datasets/mini_pos/', pre_transform=FaceToEdge())
samples = len(dataset)
if p.shuffle_dataset:
    dataset = dataset.shuffle()
n_features = dataset.get(0).x.shape[1]

model = TwoConv(n_features).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=p.weight_decay)

writer = SummaryWriter(comment='model:{}_lr:{}_lr_decay:{}'.format(
                       p.version,
                       p.learn_rate,
                       p.lr_decay))

cutoff = int(np.floor(samples*(1-p.validation_split)))
train_dataset = dataset[:cutoff]
test_dataset = dataset[cutoff:]


train_loader = DataLoader(train_dataset, shuffle=p.shuffle_dataset, batch_size=p.batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=len(test_dataset))

test_data = next(iter(test_loader))
test_labels = test_data.y.to(device)


# previous loss stored for adaptive learning rate.
tr_loss = 1
prev_loss = 1
prev_prev_loss = 1

# DEV: only one batch!
data = next(iter(train_loader))

for epoch in range(1, epochs+1):
    # rotate the structures between epochs

    model.train()
    first_batch_labels = torch.Tensor()
    pred = torch.Tensor()
    tr_weights = torch.Tensor()

    if prev_loss < tr_loss and prev_prev_loss < tr_loss:  # adaptive learning rate.
        for g in optimizer.param_groups:
            lr = lr*p.lr_decay  # Does this add overhead?
            g['lr'] = lr
    prev_prev_loss = prev_loss
    prev_loss = tr_loss


# ------- DEV: 1 batch run --------------------------
    batch_n = 0
#    for batch_n, data in enumerate(train_loader):
    optimizer.zero_grad()
    x, edge_index = data.x.to(device), data.edge_index.to(device)
    labels = data.y.to(device)
    weights = generate_weights(labels)
    tr_loss, out = model(x, edge_index, labels, weights)
    tr_loss.backward()
    optimizer.step()
    if batch_n == 0:
        tr_weights = weights
        first_batch_labels = data.y.clone().detach().to(device)
        pred = out.clone().detach().round().to(device)

    print("---- Round {}: loss={:.4f} lr:{:.6f}"
          .format(epoch, tr_loss, optimizer.param_groups[0]['lr']))

    #  --------------  REPORTING ------------------------------------

    train_precision = precision(pred, first_batch_labels, 2)[1].item()
    train_recall = recall(pred, first_batch_labels, 2)[1].item()
    train_f1 = f1_score(pred, first_batch_labels, 2)[1].item()
    roc_auc = roc_auc_score(first_batch_labels.cpu(), pred.cpu(), sample_weight=tr_weights.cpu())

    model.eval()
    x, edge_index = test_data.x.to(device), test_data.edge_index.to(device)
    labels = test_data.y.to(device)
    te_weights = generate_weights(labels)
    te_loss, out = model(x, edge_index, labels, te_weights)
    pred = out.detach().round().to(device)

    test_precision = precision(pred, test_labels, 2)[1].item()
    test_recall = recall(pred, test_labels, 2)[1].item()
    test_f1 = f1_score(pred, test_labels, 2)[1].item()
    roc_auc_te = roc_auc_score(labels.cpu(), pred.cpu(), sample_weight=te_weights.cpu())

    writer.add_scalars('Recall', {'train': train_recall,
                                  'test': test_recall}, epoch)
    writer.add_scalars('Precision', {'train': train_precision,
                                     'test': test_precision}, epoch)
    writer.add_scalars('F1_score', {'train': train_f1,
                                    'test': test_f1}, epoch)
    writer.add_scalars('Loss', {'train': tr_loss,
                                'test': te_loss}, epoch)
    writer.add_scalars('ROC AUC', {'train': roc_auc,
                                   'test': roc_auc_te}, epoch)
    writer.add_scalar('learning rate', lr, epoch)
    writer.add_histogram('Layer 1 weight gradients', model.conv1.weight.grad, epoch+1)
    '''
    writer.add_histogram('Layer 1 weights', model.conv1.weight, epoch+1)
    writer.add_histogram('Layer 1 bias', model.conv1.bias, epoch+1)


    writer.add_histogram('Layer 2 weights', model.lin1.weight, epoch+1)
    writer.add_histogram('Layer 2 bias', model.lin1.bias, epoch+1)
    writer.add_histogram('Layer 2 weight gradients', model.lin1.weight.grad, epoch+1)
    '''

writer.close()

now = datetime.datetime.now().strftime('%y%m%d%H%M')
torch.save(model.state_dict(), 'models/{}_{}.pt'.format(str(model).split('(')[0], now))
