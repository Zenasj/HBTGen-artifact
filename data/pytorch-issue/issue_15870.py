import torch
import torch.nn.functional as F

In [13]: F.nll_loss(torch.log(torch.tensor([[0.5, 0.25, 0.25], [0.25, 0.5, 0.25]])), torch.tensor([0, 1]), reduction='sum')
Out[13]: tensor(1.3863)

In [14]: F.nll_loss(torch.log(torch.tensor([[0.5, 0.25, 0.25]])), torch.tensor([0]), reduction='sum')
Out[14]: tensor(0.6931)

tensor([]) # reduction='none'
tensor(0.) # reduction='sum'
tensor(nan) # reduction='mean'