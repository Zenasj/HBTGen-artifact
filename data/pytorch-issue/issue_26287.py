import torch
import torch.nn as nn

x = torch.randn(2, dtype=torch.float32)
softmax = torch.nn.Softmax(dim=0)
softmax(x.to_mkldnn())