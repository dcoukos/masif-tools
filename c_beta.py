import torch
import numpy as np
from torch_geometric.data import DataLoader
from torch_geometric.transforms import FaceToEdge, TwoHop, RandomRotate, Compose, Center
from torch_geometric.nn import DataParallel
from torch_geometric.utils import precision, recall, f1_score
from dataset import StructuresDataset
from transforms import *
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import roc_auc_score
from utils import generate_weights, generate_example_surfaces, make_model_directory
import params as p
from statistics import mean
import torch.nn.functional as F
from tqdm import tqdm

'''
Should structure data in a similar way, but simply use a different label type.
Training can be performed initially by-structure, just to get the code up and running.
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
modelpath = make_model_directory('c_beta_models')


# ---- Importing and structuring Datasets and Model ----
print('Importing structures.')
# Remember!!! Shape Index can only be computed on local. Add other transforms after
# Pre_tranform step to not contaminate the data.
trainset = StructuresDataset(root='./datasets/res_train/', prefilter=None,
                             pre_transform=Compose((FaceAttributes(), NodeCurvature(),
                                                    FaceToEdge(), TwoHop())))

# Define transform in epoch, so that rotation occurs around Δ axis every time.
len(trainset)

if p.shuffle_dataset:
    trainset = trainset.shuffle()
n_features = trainset.get(0).x.shape[1]
