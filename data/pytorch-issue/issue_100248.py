import torch

a = torch.tensor(1., requires_grad=True)
b = torch.tensor(2., requires_grad=True)
out1, out2 = torch._foreach_sigmoid([a, b])
out1.backward()