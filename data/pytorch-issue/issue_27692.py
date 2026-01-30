import torch.nn as nn

import torch
# NOTE that in this situation, when N is less than 10, the results are usually
# equal or similar, and the difference becomes larger when N is increased
a = torch.rand(12, 3, 1024, 1024)
print(torch.nn.L1Loss()(a, torch.zeros_like(a)))
print(a.abs().mean())