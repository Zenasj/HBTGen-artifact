import torch
import torch.nn as nn
import numpy as np

N, C = 1, 1
criterion = nn.NLLLoss()
model = nn.Conv2d(C, 3, (3, 3)).cuda()
inp = Variable(torch.randn(N, 1, 3, 3)).cuda()
pred = model(inp)
target = Variable(torch.from_numpy(np.array([3]))).view(1, 1, 1).cuda()
criterion(pred, target)