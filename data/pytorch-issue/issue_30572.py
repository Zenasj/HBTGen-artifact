import torch.nn as nn

import torch
import torch.nn.functional as F

a = torch.rand(2, 2, 2, 2, requires_grad=True).half().cuda()
b = F.softmax(a, dim=-1, dtype=torch.float32)[0, 0, 0, 0]
c = F.softmax(a, dim=1, dtype=torch.float32)[0, 0, 0, 0]

b.backward(retain_graph=True)
c.backward(retain_graph=True)