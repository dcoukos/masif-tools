import torch
import numpy as np
from torch_geometric.data import DataLoader
from models import ThreeConv, SixConv, SixConvPassThrough, SixConvPT_LFC, SixConvResidual
from torch_geometric.transforms import FaceToEdge
from torch_geometric.utils import precision, recall, f1_score
from torch_geometric.nn import DataParallel
from dataset import MiniStructures, Structures
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from utils import generate_weights
import datetime
import params as p
from statistics import mean
import torch.functional as F
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

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# reproducibility
torch.manual_seed(p.random_seed)
np.random.seed(p.random_seed)

lr = p.learn_rate

if str(device) == 'cuda:0':
    epochs = p.epochs
else:
    epochs = 20

dataset = Structures(root='./datasets/{}/'.format(p.dataset),
                     pre_transform=FaceToEdge(), prefix=p.dataset)
samples = len(dataset)
if p.shuffle_dataset:
    dataset = dataset.shuffle()
n_features = dataset.get(0).x.shape[1]

model = SixConvResidual(n_features, heads=1, dropout=p.dropout).to(device)
model = DataParallel(model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=p.weight_decay)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
                                                       factor=p.lr_decay,
                                                       patience=p.patience)

writer = SummaryWriter(comment='model:{}_lr:{}_lr_decay:{}'.format(
                       p.version,
                       lr,
                       p.lr_decay))

cutoff = int(np.floor(samples*(1-p.validation_split)))
train_dataset = dataset[:cutoff]
test_dataset = dataset[cutoff:]


train_loader = DataLoader(train_dataset, shuffle=p.shuffle_dataset, batch_size=p.batch_size)
test_loader = DataLoader(test_dataset, shuffle=False, batch_size=p.test_batch_size)

intermediate_lr = p.intermediate_learn_rate
prev_lr = lr
steps_down = 0

for epoch in range(1, epochs+1):
    # rotate the structures between epochs

    # Reboost the learning rate
    learn_rate = optimizer.param_groups[0]['lr']
    if prev_lr > learn_rate:
        steps_down += 1
        prev_lr = learn_rate
    if steps_down == 5:
        for group in optimizer.param_groups:
            group['lr'] = intermediate_lr
            intermediate_lr *= p.lr_cap_decay
        steps_down = 0

    model.train()
    first_batch_labels = torch.Tensor()
    pred = torch.Tensor()
    tr_weights = torch.Tensor()
    loss = []

    for batch_n, data in enumerate(train_loader):
        optimizer.zero_grad()
        x, edge_index = data.x, data.edge_index
        out = model(x, edge_index)
        labels = data.y.to(out.device)
        weights = generate_weights(labels)
        tr_loss = F.binary_cross_entropy(x, target=labels, weight=generate_weights())
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
    '''
    train_precision = precision(pred, first_batch_labels, 2)[1].item()
    train_recall = recall(pred, first_batch_labels, 2)[1].item()
    train_f1 = f1_score(pred, first_batch_labels, 2)[1].item()
    '''
    roc_auc = roc_auc_score(first_batch_labels.cpu(), pred.cpu(), sample_weight=tr_weights.cpu())

    model.eval()
    cum_pred = torch.Tensor()
    cum_labels = torch.Tensor()
    te_weights = torch.Tensor()
    for batch_n, data in enumerate(test_loader):
        x, edge_index = data.x, data.edge_index
        out = model(x, edge_index)
        labels = data.y.to(out.device)
        weights = generate_weights(labels)
        te_loss = F.binary_cross_entropy(x, target=labels, weight=generate_weights())
        loss.append(te_loss.detach().item())
        pred = out.detach().round().to(device)
        cum_labels = torch.cat((cum_labels, labels.clone().detach()), dim=0)
        cum_pred = torch.cat((cum_pred, pred.clone().detach()), dim=0)
        te_weights = torch.cat((te_weights, weights.clone().detach()), dim=0)
    '''
    test_precision = precision(cum_pred, cum_labels, 2)[1].item()
    test_recall = recall(cum_pred, cum_labels, 2)[1].item()
    test_f1 = f1_score(cum_pred, cum_labels, 2)[1].item()
    '''
    roc_auc_te = roc_auc_score(cum_labels.cpu(), cum_pred.cpu(), sample_weight=te_weights.cpu())

    '''
    writer.add_scalars('Recall', {'train': train_recall,
                                  'test': test_recall}, epoch)
    writer.add_scalars('Precision', {'train': train_precision,
                                     'test': test_precision}, epoch)
    writer.add_scalars('F1_score', {'train': train_f1,
                                    'test': test_f1}, epoch)
    '''
    writer.add_scalars('Loss', {'train': tr_loss,
                                'test': te_loss}, epoch)
    writer.add_scalars('ROC AUC', {'train': roc_auc,
                                   'test': roc_auc_te}, epoch)
    writer.add_scalar('learning rate', learn_rate, epoch)

# Observing the gradient for model v.10
# looking for dead neurons/dying,exploding gradient
    '''
    writer.add_histogram('Layer 1 weight gradients', model.conv1.weight.grad, epoch+1)
    writer.add_histogram('Layer 2 weight gradients', model.conv2.weight.grad, epoch+1)
    writer.add_histogram('Layer 3 weight gradients', model.conv3.weight.grad, epoch+1)
    writer.add_histogram('Layer 4 weight gradients', model.conv4.weight.grad, epoch+1)
    writer.add_histogram('Layer 5 weight gradients', model.conv5.weight.grad, epoch+1)
    writer.add_histogram('Layer 6 weight gradients', model.conv6.weight.grad, epoch+1)
    writer.add_histogram('Layer 7 weight gradients', model.lin1.weight.grad, epoch+1)
    writer.add_histogram('Layer 8 weight gradients', model.lin2.weight.grad, epoch+1)
    writer.add_histogram('Output layer weight gradients', model.out.weight.grad, epoch+1)
    writer.add_histogram('Layer 1 weights', model.conv1.weight, epoch+1)
    writer.add_histogram('Layer 2 weights', model.conv2.weight, epoch+1)
    writer.add_histogram('Layer 3 weights', model.conv3.weight, epoch+1)

    writer.add_histogram('Layer 4 weights', model.conv4.weight, epoch+1)
    writer.add_histogram('Layer 5 weights', model.conv5.weight, epoch+1)

    writer.add_histogram('Layer 6 weights', model.conv6.weight, epoch+1)
    writer.add_histogram('Layer 7 weights', model.lin1.weight, epoch+1)
    writer.add_histogram('Layer 8 weights', model.lin2.weight, epoch+1)
    writer.add_histogram('Output layer weights', model.out.weight, epoch+1)
    '''
    scheduler.step(loss)
writer.close()

now = datetime.datetime.now().strftime('%y%m%d%H%M')
torch.save(model.state_dict(), 'models/{}_{}.pt'.format(str(model).split('(')[0], now))
