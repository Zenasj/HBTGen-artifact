import torch
import torch.nn as nn
lstm = nn.LSTM(3, 3, bidirectional=1)
out = lstm(torch.randn(1, 1, 3))