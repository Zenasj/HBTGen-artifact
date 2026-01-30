import torch

obs.dtype
Out[28]: dtype('float32')
obs.shape
Out[29]: (54, 1, 64, 64)
a.shape
Out[30]: torch.Size([2])
obs.shape
Out[31]: (54, 1, 64, 64)
b = torch.tensor([3])
obs[b].shape
Out[33]: (1, 64, 64)
a = torch.tensor([3, 3])
obs[a].shape
Out[35]: (2, 1, 64, 64)