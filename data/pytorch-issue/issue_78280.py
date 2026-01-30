import torch.nn as nn

import torch
unpool = torch.nn.MaxUnpool2d((2, 2)).to('cuda')
output = torch.rand((1, 3, 4, 5), dtype=torch.float32, device='cuda')
indices = torch.zeros((1, 3, 4, 5), dtype=torch.int64, device='cuda')
indices.flatten()[0] = -1
unpool(output, indices)

...
res = unpool(output, indices)
print(res)