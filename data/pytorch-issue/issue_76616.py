import torch.nn as nn

import torch
groups = 3
input_tensor = torch.rand([0, 9, 4, 4])

torch.nn.ChannelShuffle(groups)(input_tensor)
# floating point exception