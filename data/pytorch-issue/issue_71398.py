import torch
x = torch.Generator().manual_seed(2)
import copy
copy.deepcopy(x)