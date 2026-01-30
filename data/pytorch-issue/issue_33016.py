import torch.nn as nn

torch.nn.LSTM(5, 5).cuda()(torch.randn(5,5,5, device="cuda"))

import torch
torch.backends.cudnn.lib is None