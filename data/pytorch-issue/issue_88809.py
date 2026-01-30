import torch.nn as nn

py
import torch

input = torch.randn(3, requires_grad=True).cuda()
target = torch.randint(low=0, high=3, size=(3,)).long().cuda()
weight = torch.rand(3, requires_grad=True).cuda()
loss = torch.nn.functional.nll_loss(input, target)
loss = (torch.sum((weight * loss)) * 0.25)
with torch.no_grad():
    input.requires_grad_(True)
    weight = weight.detach_()
    loss.backward()