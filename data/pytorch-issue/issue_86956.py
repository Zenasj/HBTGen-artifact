import torch

torch.arange(4**3).reshape(4, 4, 4).permute((2, 0, 1))[1:,::2]

x[-1,-1,-1].storage_offset()-x[0,0,0].storage_offset() == x.numel()