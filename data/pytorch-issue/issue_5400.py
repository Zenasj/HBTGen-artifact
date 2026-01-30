import torch.nn as nn

import torch
from torch import nn
from torch.autograd import Variable

m = nn.Sequential(
    nn.Conv1d(1, 8, kernel_size=10, stride=1),
    nn.MaxPool1d(2496)
)

a = torch.randn(256, 1, 2500)
m(Variable(a))