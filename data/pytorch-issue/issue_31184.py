import torch
a=torch.zeros(256)
a.unflatten(-1, [('x', 2), ('y',128)])