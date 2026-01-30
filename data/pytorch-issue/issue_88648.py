import torch.nn as nn

py
import torch

torch.manual_seed(420)
input = torch.randn(3, 5, requires_grad=True).cuda()
target = torch.tensor([1, 0, 5]).cuda()

loss = torch.nn.MultiMarginLoss()
output = loss(input, target)

print(output)
# CUDA
# tensor(0.7524, device='cuda:0', grad_fn=<MultiMarginLossBackward0>)

py
import torch

torch.manual_seed(420)
input = torch.randn(3, 5, requires_grad=True)
target = torch.tensor([1, 0, 5])

loss = torch.nn.MultiMarginLoss()
output = loss(input, target)
# CPU
# RuntimeError: target out of range