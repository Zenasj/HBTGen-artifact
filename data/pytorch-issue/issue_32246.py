import torch

@torch.jit.script
def method():
    mask1 = torch.tensor([True, True, True])
    mask2 = torch.zeros(mask1.shape[0], dtype=torch.int8)
    ixs = torch.arange(mask1.shape[0])[mask1]

    mask2[ixs] = -1

@torch.jit.script
def method():
    mask1 = torch.tensor([True, True, True])
    mask2 = torch.zeros(mask1.shape[0], dtype=torch.int8)
    ixs = torch.arange(mask1.shape[0])[mask1]

    mask2.index_put((ixs, ), torch.tensor(-1))

@torch.jit.script
def method():
    mask1 = torch.tensor([True, True, True])
    mask2 = torch.zeros(mask1.shape[0], dtype=torch.int8)
    ixs = torch.arange(mask1.shape[0])[mask1]

    mask2[ixs] = torch.tensor(-1)