import torch.nn as nn

import torch
groups = 2
m = torch.nn.ChannelShuffle(groups=groups)
input_tensor = torch.rand([6, 0, 6], dtype=torch.float32).to('cuda')
m.to('cuda')
m(input_tensor)