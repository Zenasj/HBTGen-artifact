import torch
from torch.utils.data import TensorDataset, ConcatDataset

ds = ConcatDataset([TensorDataset(torch.randn(3,2,3)), TensorDataset(torch.randn(7,2,3))])

slice = ds[2:6]