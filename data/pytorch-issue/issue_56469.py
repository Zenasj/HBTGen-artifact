import torch
import torch.nn as nn

input = torch.tensor([[0., 1., 3.], [2., 4., 0.]], requires_grad=True)
target = torch.tensor([[1., 4., 2.], [-1., 2., 3.]])
var = 2*torch.ones(size=(2, 3), requires_grad=True)

loss = torch.nn.GaussianNLLLoss(reduction='none')

print(loss(input, target, var)) 
# Gives tensor([3.7897, 6.5397], grad_fn=<MulBackward0>. This has size (2).

print(loss(input, target, var)) 
# Gives tensor([[0.5966, 2.5966, 0.5966], [2.5966, 1.3466, 2.5966]], grad_fn=<MulBackward0>)
# This has the expected size, (2, 3).

print(loss(input, target, var).sum(dim=1))
# Gives tensor([3.7897, 6.5397], grad_fn=<SumBackward1>.