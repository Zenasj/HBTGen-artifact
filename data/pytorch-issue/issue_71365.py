import torch.nn as nn

import torch

seq = torch.nn.utils.rnn.pad_sequence(torch.tensor([[[ 7,  6]], [[-7, -1]]]))