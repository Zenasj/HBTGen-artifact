import torch
import torch.nn as nn

input = torch.randn((20, 30, 32, 32)).to("cuda:0")
fc = nn.Linear(5, 3).to("cuda:0")
print(fc(input).shape)