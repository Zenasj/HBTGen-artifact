import torch.nn as nn
m = nn.AdaptiveMaxPool1d(1)
inputs = torch.rand([0,1,0])
m(inputs)