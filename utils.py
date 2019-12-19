import torch


def perf_measure(pred, labels):
    '''Calculates and returns performance metrics from two tensors when doing binary
        classification'''

    A = pred == labels
    B = pred == torch.ones(size=pred.shape)

    TP = (A * B).sum().item()
    FP = (~A * B).sum().item()
    TN = (A * ~B).sum().item()
    FN = (~A * ~B).sum().item()

    return (TP, FP, TN, FN)
