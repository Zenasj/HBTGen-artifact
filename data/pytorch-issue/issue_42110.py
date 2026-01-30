import torch.nn as nn

import torch
import torch.nn.functional as F

lprobs = torch.log(torch.Tensor([[0.1, 0.9]]))
target = torch.Tensor((3,)).long()

try:
    F.nll_loss(lprobs.cuda(), target.cuda())
except IndexError:
    print("Raised GPU error")

try:
    F.nll_loss(lprobs, target)
except IndexError:
    print("Raised CPU error")