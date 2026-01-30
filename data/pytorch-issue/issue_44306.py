import torch
import torch.nn as nn
from torch.autograd import Variable

src_vocab = 5

def Loss(x, tgt):
    criterion = nn.KLDivLoss(reduction='batchmean')
    padding_idx = 0
    confidence = 0.9
    true_dist = None

    true_dist = torch.zeros_like(x) + ((1.0 - confidence) / (src_vocab - 2))
    true_dist = x.data.clone()
    true_dist.fill_((1.0 - confidence) / (src_vocab - 2))
    true_dist.scatter_(1, tgt.data.unsqueeze(1), confidence)
    true_dist[:, padding_idx] = 0
    mask = torch.nonzero(tgt.data == padding_idx,as_tuple=False)
    if mask.dim() > 0:
        true_dist.index_fill_(0, mask.squeeze(), 0.0)
    true_dist = true_dist
    return criterion(x, Variable(true_dist, requires_grad=False))

print(Loss(torch.tensor([
                [0.1, 0.1, 0.1, 0.8, 0.1],
                [0.1, 0.1, 0.1, 0.8, 0.1]
            ]), torch.tensor([[3],[3]])))