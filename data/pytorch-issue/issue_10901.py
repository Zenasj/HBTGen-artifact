import torch.nn as nn

t = th.nn.Parameter(th.tensor([1.])).cuda()
t.grad = th.tensor([1.]).cuda()