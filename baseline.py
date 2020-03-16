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
                                             FaceToEdge(), TwoHop())),
                      transform=Compose((AddShapeIndex(), AddRandomFeature())))
validset = Structures(root='./datasets/masif_site_test/',
                      pre_transform=Compose((FaceAttributes(), NodeCurvature(),
                                             FaceToEdge(), TwoHop())),
                      transform=Compose((AddShapeIndex(), AddRandomFeature())))

if p.shuffle_dataset:
    trainset = trainset.shuffle()
n_features = trainset.get(0).x.shape[1]

# ---- Import previous model to allow deep network to train -------------
print('Setting up model...')
prev_model = torch.load('./models/Feb27_11:07_exp1_10conv-elec+SI/best.pt', map_location=cpu)
model = p.model_type(5, heads=p.heads).to(cpu)

conv1_weights = prev_model['conv1.weight']
extra_row = torch.ones(1, 64)*.00000001
conv1_weights = torch.cat((conv1_weights, extra_row), dim=0)
prev_model['conv1.weight'] = conv1_weights

conv1_u = prev_model['conv1.u']
conv1_u = torch.cat((conv1_u, torch.tensor([-0.05, -0.05, 0.05, 0.05]).view(1, 4)), dim=0)
prev_model['conv1.u'] = conv1_u
model.load_state_dict(prev_model)
model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=learn_rate, weight_decay=p.weight_decay)

writer = SummaryWriter(comment='model:{}_lr:{}_shuffle:{}_seed:{}'.format(
                       p.version,
                       learn_rate,
                       p.shuffle_dataset,
                       p.random_seed))


max_roc_auc = 0

# ---- Training ----
print('Training...')
for epoch in range(1, epochs+1):

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

    roc_auc_te = roc_auc_score(cum_labels.cpu(), cum_pred.cpu())
    writer.add_scalars('Loss', {'train': tr_loss,
                                'test': te_loss}, epoch)
    writer.add_scalars('ROC AUC', {'train': roc_auc,
                                   'test': roc_auc_te}, epoch)
    writer.add_scalar('learning rate', learn_rate, epoch)

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
