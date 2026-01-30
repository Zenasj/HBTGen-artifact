import torch.nn as nn

import torch
torch._C._nn.rrelu_with_noise_(torch.rand([9]), lower=-1, noise=torch.rand([3]), training=True)