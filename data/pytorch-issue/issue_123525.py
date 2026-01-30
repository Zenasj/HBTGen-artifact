input = torch.randn(5, 10)
m = torch.nn.InstanceNorm1d(num_features=5, eps=2018915490, affine=True)
output = m(input)
print(output)

import torch
import torch.nn as nn
import torch.nn.utils as nn_utils

input = torch.randn(3, 10)
linear = nn.Linear(in_features=10, out_features=5)
linear = nn_utils.spectral_norm(linear, eps=2018915490)
output = linear(input)
print(output)