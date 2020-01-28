import torch
import numpy as np
from torch_geometric.data import DataListLoader
from torch_geometric.transforms import FaceToEdge, TwoHop, RandomRotate, Compose, Center
from torch_geometric.nn import DataParallel
from dataset import Structures, remove_pos_data, add_pos_data
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from utils import generate_weights, generate_example_surfaces, make_model_directory
import params as p
from statistics import mean
import torch.nn.functional as F
'''
baseline.py implements a baseline model. Experiment using pytorch-geometric
    and FeaStNet.

Focus on just 1 conv small conv layer first! Learn on 1 structure, then 10 structures.
Then increase model complexity to accomodate the increase in data.
Ignore test metrics for now.
'''

# --- Parameter setting -----
if p.suppress_warnings:
    import warnings
    warnings.filterwarnings("ignore")

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# reproducibility
torch.manual_seed(p.random_seed)
np.random.seed(p.random_seed)
learn_rate = p.learn_rate
modelpath = make_model_directory()

if str(device) == 'cuda:0':
    epochs = p.epochs
else:
    epochs = 20

# ---- Importing and structuring Datasets and Model ----
if p.twohop is True:
    print("Adding two-hop edges to data graphs")
    converter = TwoHop()
else:
    converter = None

trainset = Structures(root='./datasets/{}_train/'.format(p.dataset),
                      pre_transform=Compose(Center(), FaceToEdge()), transform=converter,
                      prefix=p.dataset)
samples = len(trainset)
cutoff = int(np.floor(samples*(1-p.validation_split)))
trainaset = trainset[:cutoff]
validset = trainset[cutoff:]

if p.shuffle_dataset:
    trainset = trainset.shuffle()
n_features = trainset.get(0).x.shape[1]
print('Setting up model...')
model = p.model_type(n_features, heads=p.heads, dropout=p.dropout).to(device)
model = DataParallel(model).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=p.weight_decay)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                       factor=p.lr_decay,
#                                                       patience=p.patience)

writer = SummaryWriter(comment='model:{}_lr:{}_lr_decay:{}_shuffle:{}_seed:{}'.format(
                       p.version,
                       learn_rate,
                       p.lr_decay,
                       p.shuffle_dataset,
                       p.random_seed))


val_loader = DataListLoader(validset, shuffle=False, batch_size=p.test_batch_size)
axes = [0, 1, 2]
max_roc_auc = 0

# ---- Training ----
print('Training...')
for epoch in range(1, epochs+1):
    # rotate the structures between epochs

    if 'pos' in p.dataset:  # Is there positional data in the features?
        remove_pos_data(trainset)
        rotation_axis = axes[epoch % 3]  # only for structural data.
        trainset.transform = RandomRotate(90, axis=rotation_axis)
        add_pos_data(trainset)
    train_loader = DataListLoader(trainset, shuffle=p.shuffle_dataset, batch_size=p.batch_size)

    learn_rate = optimizer.param_groups[0]['lr']  # for when it may be modified during run
    # rotator()
    model.train()
    first_batch_labels = torch.Tensor()
    pred = torch.Tensor()
    tr_weights = torch.Tensor()
    loss = []

    for batch_n, datalist in enumerate(train_loader):
        optimizer.zero_grad()
        out = model(datalist)
        labels = torch.cat([data.y for data in datalist]).to(out.device)
        weights = generate_weights(labels).to(out.device)
        tr_loss = F.binary_cross_entropy(out, target=labels, weight=weights)
        loss.append(tr_loss.detach().item())
        tr_loss.backward()
        optimizer.step()
        if batch_n == 0:
            tr_weights = weights
            first_batch_labels = labels.clone().detach().to(device)
            pred = out.clone().detach().round().to(device)

    loss = mean(loss)
    print("---- Round {}: loss={:.4f} lr:{:.6f}"
          .format(epoch, loss, learn_rate))

    #  --------------  REPORTING ------------------------------------
    roc_auc = roc_auc_score(first_batch_labels.cpu(), pred.cpu())

    model.eval()
    cum_pred = torch.Tensor().to(device)
    cum_labels = torch.Tensor().to(device)
    te_weights = torch.Tensor().to(device)
    for batch_n, datalist in enumerate(val_loader):
        out = model(datalist)
        labels = torch.cat([data.y for data in datalist]).to(out.device)
        weights = generate_weights(labels).to(out.device)
        te_loss = F.binary_cross_entropy(out, target=labels, weight=generate_weights(labels))
        pred = out.detach().round().to(device)
        cum_labels = torch.cat((cum_labels, labels.clone().detach()), dim=0)
        cum_pred = torch.cat((cum_pred, pred.clone().detach()), dim=0)
        te_weights = torch.cat((te_weights, weights.clone().detach()), dim=0)

    roc_auc_te = roc_auc_score(cum_labels.cpu(), cum_pred.cpu())
    writer.add_scalars('Loss', {'train': tr_loss,
                                'test': te_loss}, epoch)
    writer.add_scalars('ROC AUC', {'train': roc_auc,
                                   'test': roc_auc_te}, epoch)
    writer.add_scalar('learning rate', learn_rate, epoch)

    if epoch % 20 == 0:
        writer.add_histogram('Layer 1 weight gradients', model.module.conv1.weight.grad, epoch+1)
        writer.add_histogram('Layer 2 weight gradients', model.module.conv2.weight.grad, epoch+1)
        writer.add_histogram('Layer 3 weight gradients', model.module.conv3.weight.grad, epoch+1)
        writer.add_histogram('Layer 4 weight gradients', model.module.conv4.weight.grad, epoch+1)
        writer.add_histogram('Layer 5 weight gradients', model.module.conv5.weight.grad, epoch+1)
        writer.add_histogram('Layer 6 weight gradients', model.module.conv6.weight.grad, epoch+1)
        writer.add_histogram('Layer 7 weight gradients', model.module.lin1.weight.grad, epoch+1)
        writer.add_histogram('Layer 8 weight gradients', model.module.lin2.weight.grad, epoch+1)
    # scheduler.step(loss)
    if roc_auc_te > max_roc_auc:
        max_roc_auc = roc_auc_te
        path = './{}/best.pt'.format(modelpath)
        with open(path, 'w+'):
            torch.save(model.module.state_dict(), path)
    if epoch % 200 == 0:
        path = './{}/epoch_{}.pt'.format(modelpath, epoch)
        with open(path, 'a+'):
            torch.save(model.module.state_dict(), path)
writer.close()


# modify to generate TEST surfaces.
# generate_example_surfaces(p.model_type, path, n_examples=8)
