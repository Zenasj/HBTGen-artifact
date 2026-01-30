import torch
torch._foreach_add([torch.randn(5)], [torch.randn(4), torch.randn(3)])