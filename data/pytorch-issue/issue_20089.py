import torch
import torch.nn as nn

# wrong
self.w = nn.Parameter(torch.ones(10)).cuda()

# why is it wrong? it is more like:
x = nn.Parameter(torch.ones(10))
self.w = x.cuda() # non-leaf variable

# correct
self.w = nn.Parameter(torch.ones(10, device='cuda'))