import torch

x = torch.nested_tensor([torch.rand(1,2,requires_grad=True), torch.rand(2,2,requires_grad=True)])
x[0]