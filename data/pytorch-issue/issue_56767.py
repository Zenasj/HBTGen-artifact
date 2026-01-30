import torch.nn as nn
import math

3
stdv = 1.0 / math.sqrt(self.hidden_size)

3
import torch

i_size = 10
h_size = 0
cell = torch.nn.GRUCell(i_size, h_size)

i = torch.rand(32, i_size)
h = torch.rand(32, h_size)
h_next = cell(i, h)

3
stdv = 1 / math.sqrt(self.hidden_size) if self.hidden_size > 0 else 0