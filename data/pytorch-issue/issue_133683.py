import torch.nn as nn

import torch
from torch import nn

my_tensor = torch.tensor([-1., 0., 1.])

tanh = nn.Tanh(3, 5) # Error

tanh = nn.Tanh(num1=3, num2=5) # Error