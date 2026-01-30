import torch.nn as nn

import torch

x = torch.zeros((5, 0, 3)).cuda()
gru = torch.nn.GRU(input_size=3, hidden_size=4).to("cuda")
gru(x)