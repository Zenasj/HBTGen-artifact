import torch.nn as nn

python
import torch
A = torch.nn.Parameter(torch.rand(5), requires_grad=True)
B = torch.heaviside(A, torch.tensor(0.5))
B.sum().backward()  # < - Error raised