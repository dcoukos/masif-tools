import torch
import numpy as np
from torch_geometric.data import DataLoader, NeighborSampler
from torch_geometric.transforms import FaceToEdge, TwoHop, RandomRotate, Compose, Center
from dataset import StructuresDataset
from transforms import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from utils import generate_weights, make_model_directory
import params as p
from statistics import mean
import torch.nn.functional as F
from models import SageNet

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
cpu = torch.device('cpu')

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
# Remember!!! Shape Index can only be computed on local. Add other transforms after
# Pre_tranform step to not contaminate the data.
trainset = StructuresDataset(root='./datasets/named_masif_train_ds/')
validset = StructuresDataset(root='./datasets/named_masif_test_ds/')
if p.shuffle_dataset:
    trainset = trainset.shuffle()
n_features = trainset.get(0).x.shape[1]
print('Setting up model...')
model = SageNet(80).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=p.weight_decay)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min',
#                                                       factor=p.lr_decay,
#                                                       patience=p.patience)

writer = SummaryWriter(comment='model:{}_lr:{}_shuffle:{}_seed:{}'.format(
                       p.version,
                       learn_rate,
                       p.shuffle_dataset,
                       p.random_seed))


# axes = [0, 1, 2]
max_roc_auc = 0
# ---- Training ----

    model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=p.weight_decay)
# ------------ TRAINING NEW BLOCK --------------------------
    print('Training block {}'.format(model_n))
    for epoch in range(1, epochs+1):
        train_loader = DataLoader(trainset, shuffle=p.shuffle_dataset, batch_size=p.batch_size)  # redefine train_loader to use data out.
        val_loader = DataLoader(validset, shuffle=False, batch_size=p.test_batch_size)
        masked_loader = DataLoader(maskedset, shuffle=False, batch_size=p.test_batch_size)

        model.train()
        first_batch_labels = torch.Tensor()
        pred = torch.Tensor()
        loss = []
        cum_pred = torch.Tensor().to(device)
        cum_labels = torch.Tensor().to(device)
        for batch_n, batch in enumerate(train_loader):
            ns = NeighborSampler(batch, 0.5, 9)
            for data_flow in ns():
                batch = batch.to(device)
                optimizer.zero_grad()
                out, _ = SageNet(batch.x.to(device), data_flow.to(device))
                labels = batch.y.to(device)
                weights = generate_weights(labels).to(device)
                tr_loss = F.binary_cross_entropy_with_logits(out, target=labels, weight=weights)
                loss.append(tr_loss.detach().item())
                tr_loss.backward()
                optimizer.step()
                cum_labels = torch.cat((cum_labels, labels.clone().detach()), dim=0)
                cum_pred = torch.cat((cum_pred, out.clone().detach()), dim=0)

        roc_auc = roc_auc_score(cum_labels.cpu(), cum_pred.cpu())
        loss = mean(loss)

#  --------------  EVALUATION & REPORTING ------------------------------------
        with torch.no_grad():
            model.eval()
            cum_pred = torch.Tensor().to(device)
            cum_labels = torch.Tensor().to(device)
            for batch_n, batch in enumerate(val_loader):
                ns = NeighborSampler(batch, 0.5, 9)
                for data_flow in ns():
                    out, _ = model(batch.x.to(device), data_flow.to(device))
                    labels = batch.y.to(device)
                    weights = generate_weights(labels).to(device)
                    te_loss = F.binary_cross_entropy_with_logits(out, target=labels, weight=generate_weights(labels))
                    pred = out.detach().round().to(device)
                    cum_labels = torch.cat((cum_labels, labels.clone().detach()), dim=0)
                    cum_pred = torch.cat((cum_pred, pred.clone().detach()), dim=0)
            roc_auc_te = roc_auc_score(cum_labels.cpu(), cum_pred.cpu())

            cum_pred = torch.Tensor().to(device)
            cum_labels = torch.Tensor().to(device)
            for batch_n, batch in enumerate(masked_loader):
                ns = NeighborSampler(batch, 0.5, 9)
                for data_flow in ns():
                    batch = batch.to(device)
                    out, _ = model(batch.x.to(device), data_flow.to(device))
                    labels = batch.y.to(device)
                    weights = generate_weights(labels).to(device)
                    te_loss = F.binary_cross_entropy_with_logits(out, target=labels, weight=generate_weights(labels))
                    pred = out.detach().round().to(device)
                    cum_labels = torch.cat((cum_labels, labels.clone().detach()), dim=0)
                    cum_pred = torch.cat((cum_pred, pred.clone().detach()), dim=0)
            roc_auc_masked = roc_auc_score(cum_labels.cpu(), cum_pred.cpu())

            writer.add_scalars('Loss', {'train': tr_loss,
                                        'test': te_loss}, epoch)
            writer.add_scalars('ROC AUC', {'train': roc_auc,
                                           'test': roc_auc_te,
                                           'masked': roc_auc_masked}, epoch)
            writer.add_scalar('learning rate', learn_rate, epoch)

            print("---- Round {}: tr_loss={:.4f} te_roc_auc:{:.4f} lr:{:.6f}"
                  .format(epoch, loss, roc_auc_te, learn_rate))

    #   -------------- MODEL SAVING ------------------------
            if roc_auc_te > max_roc_auc:
                max_roc_auc = roc_auc_te
                path = './{}/best.pt'.format(modelpath)
                with open(path, 'w+'):
                    torch.save(model.state_dict(), path)

writer.close()
