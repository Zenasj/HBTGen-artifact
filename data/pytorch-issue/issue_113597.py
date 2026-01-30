import torch

# RuntimeError: NYI: named tensors only support CPU, CUDA or privateuseone tensors.
t = torch.empty(3, device='meta', names=['c'])

# works fine
t = torch.empty(3, device='meta')
t.names = ['c']