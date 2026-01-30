import torch.nn as nn

import torch
import torch.nn.functional as F

x = torch.tensor([])
y = torch.tensor([])

loss = F.binary_cross_entropy_with_logits

print(loss(x.to("cpu"),y.to("cpu"))) # tensor(nan)
if torch.cuda.is_available():
    print(loss(x.to("cuda"),y.to("cuda"))) # tensor(nan, device='cuda:0')
if torch.backends.mps.is_available():
    print(loss(x.to("mps"),y.to("mps"))) # tensor(0., device='mps:0')

import torch
import torch.nn.functional as F

x, y = torch.tensor([]), torch.tensor([])
loss = F.binary_cross_entropy_with_logits

print(loss(x.to('cpu'), y.to('cpu')))
print(loss(x.to('mps'), y.to('mps')))