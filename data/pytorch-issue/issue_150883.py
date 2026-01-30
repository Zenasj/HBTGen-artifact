import torch
x = torch.tensor([5.0, 6.0], requires_grad=True)
y = (x * 2).sum()
torch.autograd.backward(tensors=y, inputs=x)

import torch
x = torch.tensor(5.0, requires_grad=True)
y = x * 2
torch.autograd.backward(tensors=y, inputs=x)