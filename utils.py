import torch
from sklearn.metrics import classification_report
from dataset import MiniStructures

def perf_measure(pred, labels):
    '''Calculates and returns performance metrics from two tensors when doing binary
        classification'''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    A = pred == labels
    B = pred == torch.ones(size=pred.shape).to(device)

    TP = (A * B).sum().item()
    FP = (~A * B).sum().item()
    TN = (A * ~B).sum().item()
    FN = (~A * ~B).sum().item()

    return (TP, FP, TN, FN)


def stats(pred, labels):
    return classification_report(labels.cpu().detach().numpy(), pred.cpu().detach().numpy())


def generate_weights(labels):
    '''
        Takes a label tensor, and generates scoring weights for binary_cross_entropy
    '''
    example = MiniStructures()
    n_nodes = float(len(example.data.y))
    n_pos_nodes = example.data.y.sum().item()
    n_neg_nodes = n_nodes - n_pos_nodes
    ratio_neg = n_neg_nodes/n_nodes
    ratio_pos = n_pos_nodes/n_nodes

    return (labels.clone().detach()*ratio_neg + ratio_pos)
