import torch.nn as nn
import random

import numpy as np
import torch
import torch.nn.functional as F

np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

inputs = torch.randn(size=(64, 3, 32, 32))
weight = torch.randn(size=(16, 3, 3, 3))

results_gpu = F.conv2d(input=inputs.cuda(), weight=weight.cuda(), stride=1, padding=1).cpu()
results_cpu = F.conv2d(input=inputs.cpu(), weight=weight.cpu(), stride=1, padding=1)
print((results_gpu - results_cpu).abs().max())