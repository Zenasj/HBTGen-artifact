import torch.nn as nn

import torch
from torch.distributed.pipeline.sync import Pipe

fc1 = torch.nn.Linear(16, 8).cuda(0)
fc2 = torch.nn.Linear(8, 4).cuda(1)
model = torch.nn.Sequential(fc1, fc2)
model = Pipe(model, chunks=8)
input = torch.rand(16, 16).cuda(0)
output_rref = model(input)