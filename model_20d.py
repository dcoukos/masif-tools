import torch
import gc
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
from models import ThreeConvBlock

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
    epochs = int(p.epochs/20)
else:
    epochs = 20

# ---- Importing and structuring Datasets and Model ----
print('Importing structures.')
# Remember!!! Shape Index can only be computed on local. Add other transforms after
# Pre_tranform step to not contaminate the data.
dataset = Structures(root='./datasets/{}_train/'.format(p.dataset),
                      pre_transform=Compose((FaceAttributes(),
                                             NodeCurvature(), FaceToEdge(),
                                             TwoHop())),
                      transform=AddShapeIndex())


samples = len(dataset)
assert(p.validation_split < 0.3)
cutoff = int(np.floor(samples*(1-p.validation_split)))
trainset = dataset[:cutoff]
validset = dataset[cutoff:]
maskedset = validset[:int(len(validset)/2)]
validset = validset[int(len(validset)/2):]


if p.shuffle_dataset:
    trainset = trainset.shuffle()
n_features = trainset.get(0).x.shape[1]
print('Setting up model...')
models = [ThreeConvBlock(n_features=4, lin2=4, heads=p.heads).to(cpu),
          ThreeConvBlock(n_features=4, lin2=4, heads=p.heads).to(cpu),
          ThreeConvBlock(n_features=4, lin2=4, heads=p.heads).to(cpu)]

# setting up reporting
writer = SummaryWriter(comment='model:{}_lr:{}_lr_decay:{}_shuffle:{}_seed:{}'.format(
                       p.version,
                       learn_rate,
                       p.lr_decay,
                       p.shuffle_dataset,
                       p.random_seed))


optimizers = [torch.optim.Adam(models[0].parameters(), lr=learn_rate, weight_decay=p.weight_decay),
              torch.optim.Adam(models[1].parameters(), lr=learn_rate, weight_decay=p.weight_decay),
              torch.optim.Adam(models[2].parameters(), lr=learn_rate, weight_decay=p.weight_decay)]

# ------ Pre-training the datasets----------
print("Pretraining the datasets.", end='')

for epoch in range(0, 10):
    model = models[0]
    model.to(device)
    optimizer = optimizers[0]
# ------------ TRAINING NEW BLOCK --------------------------
    train_loader = DataLoader(trainset_, shuffle=p.shuffle_dataset, batch_size=p.batch_size)  # redefine train_loader to use data out.

    model.train()
    first_batch_labels = torch.Tensor()
    pred = torch.Tensor()
    loss = []

    for batch_n, batch in enumerate(train_loader):
        batch = batch.to(device)
        optimizer.zero_grad()
        out, _ = model(batch)
        labels = batch.y.to(device)
        weights = generate_weights(labels).to(device)
        tr_loss = F.binary_cross_entropy(out, target=labels, weight=weights)
        loss.append(tr_loss.detach().item())
        tr_loss.backward()
        if batch_n == 0:
            first_batch_labels = labels.clone().detach().to(device)
            pred = out.clone().detach().round().to(device)

    print('.', end='')

print()

for model in models[1:]:
    model.load_state_dict(models[0], map_location=device)

# ---- Training ----
max_roc_te = [0, 0, 0]
max_roc_masked = [0, 0, 0]

for cycle in range(0, epochs):
    trainset_ = trainset
    validset_ = validset
    maskedset_ = maskedset
    for model_n, model in enumerate(models):
        model.to(device)
        optimizer = optimizers[model_n]
    # ------------ TRAINING NEW BLOCK --------------------------
        train_loader = DataLoader(trainset_, shuffle=p.shuffle_dataset, batch_size=p.batch_size)  # redefine train_loader to use data out.
        val_loader = DataLoader(validset_, shuffle=False, batch_size=p.test_batch_size)
        masked_loader = DataLoader(maskedset_, shuffle=False, batch_size=p.test_batch_size)

        model.train()
        first_batch_labels = torch.Tensor()
        pred = torch.Tensor()
        loss = []

        for batch_n, batch in enumerate(train_loader):
            batch = batch.to(device)
            optimizer.zero_grad()
            out, _ = model(batch)
            labels = batch.y.to(device)
            weights = generate_weights(labels).to(device)
            tr_loss = F.binary_cross_entropy(out, target=labels, weight=weights)
            loss.append(tr_loss.detach().item())
            tr_loss.backward()
            if batch_n == 0:
                first_batch_labels = labels.clone().detach().to(device)
                pred = out.clone().detach().round().to(device)

        loss = mean(loss)

#  --------------  EVALUATION & REPORTING ------------------------------------
        roc_auc = roc_auc_score(first_batch_labels.cpu(), pred.cpu())
        with torch.no_grad():
            model.eval()
            cum_pred = torch.Tensor().to(device)
            cum_labels = torch.Tensor().to(device)
            for batch_n, batch in enumerate(val_loader):
                batch = batch.to(device)
                out, _ = model(batch)
                labels = batch.y.to(device)
                weights = generate_weights(labels).to(device)
                te_loss = F.binary_cross_entropy(out, target=labels, weight=generate_weights(labels))
                pred = out.detach().round().to(device)
                cum_labels = torch.cat((cum_labels, labels.clone().detach()), dim=0)
                cum_pred = torch.cat((cum_pred, pred.clone().detach()), dim=0)
            roc_auc_te = roc_auc_score(cum_labels.cpu(), cum_pred.cpu())

            cum_pred = torch.Tensor().to(device)
            cum_labels = torch.Tensor().to(device)
            for batch_n, batch in enumerate(masked_loader):
                batch = batch.to(device)
                out, _ = model(batch)
                labels = batch.y.to(device)
                weights = generate_weights(labels).to(device)
                te_loss = F.binary_cross_entropy(out, target=labels, weight=generate_weights(labels))
                pred = out.detach().round().to(device)
                cum_labels = torch.cat((cum_labels, labels.clone().detach()), dim=0)
                cum_pred = torch.cat((cum_pred, pred.clone().detach()), dim=0)
            roc_auc_masked = roc_auc_score(cum_labels.cpu(), cum_pred.cpu())

            writer.add_scalars('Loss', {'train': tr_loss,
                                        'test': te_loss}, cycle*3+model_n)
            writer.add_scalars('ROC AUC', {'train': roc_auc,
                                           'test': roc_auc_te,
                                           'masked': roc_auc_masked}, cycle*3+model_n)
            writer.add_scalar('learning rate', learn_rate, cycle*3+model_n)

            print("---- Round {}: tr_loss={:.4f} te_roc_auc:{:.4f} lr:{:.6f}"
                  .format(cycle*3+model_n, loss, roc_auc_te, learn_rate))

    #   -------------- MODEL SAVING ------------------------
            if roc_auc_te > max_roc_te[model_n]:
                max_roc_te[model_n] = roc_auc_te
                path = './{}/best_{}.pt'.format(modelpath, model_n)
                with open(path, 'w+'):
                    torch.save(model.state_dict(), path)

            if roc_auc_masked > max_roc_masked[model_n]:
                max_roc_masked[model_n] = roc_auc_masked
                path = './{}/masked_model_{}.pt'.format(modelpath, model_n)
                with open(path, 'w+'):
                    torch.save(model.state_dict(), path)

    # ----------- Preparing features from best version of this block -------------

    with torch.no_grad():
        if model_n < len(models)-1:
            print('Preparing the best version of this model for next model input.')
            model.load_state_dict(torch.load('./{}/masked_model_{}.pt'.format(modelpath, model_n), map_location=device))
            model.eval()

            train_loader = DataLoader(trainset, shuffle=p.shuffle_dataset, batch_size=p.batch_size)  # redefine train_loader to use data out.
            val_loader = DataLoader(validset, shuffle=False, batch_size=p.test_batch_size)
            masked_loader = DataLoader(maskedset, shuffle=False, batch_size=p.test_batch_size)

            next_data = []
            for batch in train_loader:
                batch = batch.to(device)
                _, inter = model(batch)
                batch.x = batch.x + inter
                next_data += batch.to(cpu).to_data_list()
            trainset_ = next_data

            next_data = []
            for batch in val_loader:
                batch = batch.to(device)
                _, inter = model(batch)
                batch.x = batch.x + inter
                next_data += batch.to(cpu).to_data_list()
            validset_ = next_data

            next_data = []
            for batch in masked_loader:
                batch = batch.to(device)
                _, inter = model(batch)
                batch.x = batch.x + inter
                next_data += batch.to(cpu).to_data_list()
            maskedset_ = next_data

writer.close()
