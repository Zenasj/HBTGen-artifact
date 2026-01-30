import torch.nn as nn

import torch
a=torch.randn(1,208,2,2).cuda()
b=torch.nn.Conv2d(208,208,kernel_size=3,stride=2,padding=1).cuda()
print(b(a).shape)
# output: torch.Size([1, 208, 1, 1])
a=torch.randn(1,209,2,2).cuda()
b=torch.nn.Conv2d(209,209,kernel_size=3,stride=2,padding=1).cuda()
print(b(a).shape)
# RuntimeError: GET was unable to find an engine to execute this computation