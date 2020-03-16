import torch
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.transforms import FaceToEdge, TwoHop, RandomRotate, Compose, Center
from torch_geometric.nn import DataParallel
from torch_geometric.utils import precision, recall, f1_score
from dataset import Structures
from transforms import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from utils import generate_weights, generate_example_surfaces, make_model_directory
import params as p
from statistics import mean
import torch.nn.functional as F
from tqdm import tqdm


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
trainset = Structures(root='./datasets/masif_site_train/',
                      pre_transform=Compose((FaceAttributes(), NodeCurvature(),
                                             FaceToEdge(), TwoHop())))
# Define transform in epoch, so that rotation occurs around Δ axis every time.
validset = Structures(root='./datasets/masif_site_test/',
                      pre_transform=Compose((FaceAttributes(), NodeCurvature(),
                                             FaceToEdge(), TwoHop())))

if p.shuffle_dataset:
    trainset = trainset.shuffle()
n_features = trainset.get(0).x.shape[1]

# ---- Import previous model to allow deep network to train -------------

model = p.model_type(9, heads=p.heads).to(cpu)

model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=p.weight_decay)

writer = SummaryWriter(comment='model:exp2_3hops_lr:{}_shuffle:{}_seed:{}'.format(
                       learn_rate,
                       p.shuffle_dataset,
                       p.random_seed))


max_roc_auc = 0

# ---- Training ----
print('Training...')
for epoch in range(1, epochs+1):
    trainset.transform = Compose((Center(), RandomRotate(90, epoch%3), AddPositionalData(),
                                  TwoHop()))
    validset.transform = Compose((Center(), RandomRotate(90, epoch%3), AddPositionalData(),
                                TwoHop()))
    train_loader = DataLoader(trainset, shuffle=p.shuffle_dataset, batch_size=p.batch_size)
    val_loader = DataLoader(validset, shuffle=False, batch_size=p.test_batch_size)

    learn_rate = optimizer.param_groups[0]['lr']  # for when it may be modified during run
    model.train()
    pred = torch.Tensor()
    tr_weights = torch.Tensor()
    loss = []

    cum_pred = torch.Tensor().to(device)
    cum_labels = torch.Tensor().to(device)
    for batch in tqdm(train_loader, desc='Training.'):
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch)
        labels = batch.y.to(device)
        weights = generate_weights(labels).to(device)
        tr_loss = F.binary_cross_entropy(out, target=labels, weight=weights)
        loss.append(tr_loss.detach().item())
        tr_loss.backward()
        optimizer.step()
        cum_labels = torch.cat((cum_labels, labels.clone().detach()), dim=0)
        cum_pred = torch.cat((cum_pred, out.clone().detach()), dim=0)

    train_precision = precision(pred, first_batch_labels, 2)[1].item()
    train_recall = recall(pred, first_batch_labels, 2)[1].item()
    train_f1 = f1_score(pred, first_batch_labels, 2)[1].item()

    roc_auc = roc_auc_score(cum_labels.cpu(), cum_pred.cpu())
    loss = mean(loss)

    #  --------------  REPORTING ------------------------------------
    model.eval()
    cum_pred = torch.Tensor().to(device)
    cum_labels = torch.Tensor().to(device)
    te_weights = torch.Tensor().to(device)
    for batch in tqdm(val_loader, desc='Evaluating.'):
        batch = batch.to(device)
        out = model(batch)
        labels = batch.y.to(device)
        weights = generate_weights(labels).to(device)
        te_loss = F.binary_cross_entropy(out, target=labels, weight=generate_weights(labels))
        pred = out.detach().round().to(device)
        cum_labels = torch.cat((cum_labels, labels.clone().detach()), dim=0)
        cum_pred = torch.cat((cum_pred, pred.clone().detach()), dim=0)
        te_weights = torch.cat((te_weights, weights.clone().detach()), dim=0)

    test_precision = precision(cum_pred, cum_labels, 2)[1].item()
    test_recall = recall(cum_pred, cum_labels, 2)[1].item()
    test_f1 = f1_score(cum_pred, cum_labels, 2)[1].item()

    roc_auc_te = roc_auc_score(cum_labels.cpu(), cum_pred.cpu())
    writer.add_scalars('Loss', {'train': tr_loss,
                                'test': te_loss}, epoch)
    writer.add_scalars('ROC AUC', {'train': roc_auc,
                                   'test': roc_auc_te}, epoch)
    writer.add_scalar('learning rate', learn_rate, epoch)
    writer.add_scalars('Recall', {'train': train_recall,
                                  'test': test_recall}, epoch)
    writer.add_scalars('Precision', {'train': train_precision,
                                     'test': test_precision}, epoch)
    writer.add_scalars('F1_score', {'train': train_f1,
                                    'test': test_f1}, epoch)

    print("---- Round {}: tr_loss={:.4f} te_roc_auc:{:.4f} lr:{:.6f}"
          .format(epoch, loss, roc_auc_te, learn_rate))
    # scheduler.step(loss)
    if roc_auc_te > max_roc_auc:
        max_roc_auc = roc_auc_te
        path = './{}/best.pt'.format(modelpath)
        with open(path, 'w+'):
            torch.save(model.state_dict(), path)
    if epoch % 200 == 0:
        path = './{}/epoch_{}.pt'.format(modelpath, epoch)
        with open(path, 'a+'):
            torch.save(model.state_dict(), path)
writer.close()
