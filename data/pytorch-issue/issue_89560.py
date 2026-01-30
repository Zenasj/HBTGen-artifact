import torch.nn as nn

import torch
x = torch.randn(4).to('meta')
model = torch.nn.PReLU().to('meta')
model(x)