import torch.nn as nn

import torch

input = torch.randn(2, 3, 10).to_mkldnn()
weight = torch.randn(3, 3, 3).to_mkldnn()
bias = torch.randn(3).to_mkldnn()
output = torch.nn.functional.conv1d(input, weight, bias)