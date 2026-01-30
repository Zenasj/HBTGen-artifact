import torch.nn as nn

import torch
lstm = torch.nn.LSTM(1, 200, 2, batch_first=True)
input = torch.zeros(20, 36, 1).long()
lstm(input, None)