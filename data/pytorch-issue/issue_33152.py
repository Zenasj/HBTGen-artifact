import torch

torch.einsum(
    'ijk,ikl->ijl',
    torch.zeros([2, 2, 2], dtype=torch.complex128),
    torch.zeros([2, 2, 2], dtype=torch.complex128))

torch.einsum(
     'ijk,ikl->ijl',
     torch.zeros([2, 2, 2], dtype=torch.complex128, device='cuda:0'),
     torch.zeros([2, 2, 2], dtype=torch.complex128, device='cuda:0'))