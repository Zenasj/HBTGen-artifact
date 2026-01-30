import torch
import torch.nn as nn

p = torch.nn.LSTM(input_size=240, hidden_size=True)
x = torch.randn(1,15,240)
print(p(x)[0].shape)

torch.Size([1, 15, 1])

p = torch.nn.LSTM(input_size=240, hidden_size=True).to('cuda')