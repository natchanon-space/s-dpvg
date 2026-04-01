import torch
from torch import Tensor


def gini_coef(ts: Tensor):
    if (ts == 0).all():
        return 0.0
    
    n = len(ts)
    sum_pair_diff = 0.0
    for i in range(n):
        sum_pair_diff += torch.sum(torch.abs(ts - ts[i]))
    return sum_pair_diff / (2.0 * n * torch.sum(ts))


def entropy(ts: Tensor):
    n = float(len(ts))
    p = Tensor([torch.sum(ts == a)/n for a in torch.unique(ts)])
    return torch.sum(-p * torch.log2(p))
