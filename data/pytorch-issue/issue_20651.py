import torch

torch.distributed.reduce(packed, dst=0)

packed = torch.cuda.FloatTensor([])