import torch.nn as nn

import torch
xx = torch.zeros((100, 256)).cuda()
model = torch.nn.Linear(128, 2, bias=True).cuda()
model(xx)