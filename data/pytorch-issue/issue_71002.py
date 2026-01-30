import torch.nn as nn

import torch
from torch import nn
model = nn.RNN(226, 512, 1, bidirectional=False).to(torch.device(2))
model(torch.ones(1, 1024, 226, device=torch.device(2)), (torch.ones(1, 1024, 512, device=torch.device(2))))