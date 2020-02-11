import torch
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.transforms import FaceToEdge, TwoHop, RandomRotate, Compose, Center
from torch_geometric.nn import DataParallel
from dataset import Structures
from transforms import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from utils import generate_weights, generate_example_surfaces, make_model_directory
import params as p
from statistics import mean
import torch.nn.functional as F
from tqdm import tqdm

'''
Implementing Model 16a: (Model 15b + Shape index data):
    - Comment out rotations.
    - Replace transforms with those that allow adding of shape_index data.
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
print('Importing structures.')
trainset = Structures(root='./datasets/{}_train/'.format(p.dataset),
                      pre_transform=Compose((Center(), FaceAttributes(),
                                             NodeCurvature(), FaceToEdge(),
                                             TwoHop())))
_, idx = has_nan(trainset)
trainset = [trainset[i] for i in range(0, len(trainset)) if i not in idx]
samples = len(trainset)

cutoff = int(np.floor(samples*(1-p.validation_split)))
validset = trainset[cutoff:]
trainset = trainset[:cutoff]


if p.shuffle_dataset:
    trainset = trainset.shuffle()
n_features = trainset.get(0).x.shape[1]
print('Setting up model...')
model = p.model_type(4, heads=p.heads).to(device)
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


# axes = [0, 1, 2]
max_roc_auc = 0

# ---- Training ----
print('Training...')
for epoch in range(1, epochs+1):
    # rotate the structures between epochs, when using pos data.

    if 'pos' in p.dataset:  # Is there positional data in the features?
        degrees = 15
        rotation_axis = axes[epoch % 3]  # only for structural data.
        trainset.transform = Compose((RemovePositionalData(),
                                      RandomRotate(degrees, axis=rotation_axis),
                                      AddPositionalData()))

    # Using shape index data:
    trainset.transform = AddShapeIndex()
    validset.transform = AddShapeIndex()
    train_loader = DataLoader(trainset, shuffle=p.shuffle_dataset, batch_size=p.batch_size)
    val_loader = DataLoader(validset, shuffle=False, batch_size=p.test_batch_size)

    learn_rate = optimizer.param_groups[0]['lr']  # for when it may be modified during run
    model.train()
    first_batch_labels = torch.Tensor()
    pred = torch.Tensor()
    tr_weights = torch.Tensor()
    loss = []

    for batch_n, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        labels = batch.y.to(device)
        weights = generate_weights(labels).to(device)
        tr_loss = F.binary_cross_entropy_with_logits(out, target=labels, weight=weights)
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
    for batch_n, batch in enumerate(val_loader):
        batch = batch.to(device)
        out = model(batch)
        labels = batch.y.to(device)
        weights = generate_weights(labels).to(device)
        te_loss = F.binary_cross_entropy_with_logits(out, target=labels, weight=generate_weights(labels))
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
