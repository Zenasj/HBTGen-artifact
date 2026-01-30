import torch
import torch.nn as nn

m = nn.Conv2d(1, 1, (3, 1), stride=(2, 1)).cuda()
i = torch.rand(131072, 1, 1537, 1).cuda()
o = m(i)
o.sum().backward()