import torch.nn as nn

import torch
from torch import nn

my_tensor = torch.tensor([-1., 0., 1.])

softsign = nn.Softsign(3, 5) # Error

softsign = nn.Softsign(num1=3, num2=5) # Error