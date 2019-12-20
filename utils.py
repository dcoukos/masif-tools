import torch
from sklearn.metrics import classification_report


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
    return classification_report(labels.cpu(), pred.cpu())
