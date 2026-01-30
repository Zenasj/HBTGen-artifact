import torch.nn as nn

import torch

model = torch.nn.Linear(3, 1)
model.to(torch.device('cuda:0'))
model.share_memory()