import torch.nn as nn

import torch
from torch import nn
import sys

print("Python version")
print (sys.version)
print('CuDNN Version')
print(torch.backends.cudnn.version())
print('CuDNN Enabled')
print(torch.backends.cudnn.enabled)

class GRU_Amp_Bug(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.linear = nn.Linear(128, 128)
        
        self.gru = nn.GRU(
            input_size=128,
            hidden_size=128
        )
        
    def forward(self, X, h):
        X = self.linear(X)
        X, h_n = self.gru(X, h)
        return X


model = GRU_Amp_Bug().to(torch.cuda.current_device())

input_X = torch.rand((1, 1, 128)).to(torch.cuda.current_device())

input_h = torch.rand((1, 1, 128)).to(torch.cuda.current_device())

scaler = torch.cuda.amp.GradScaler()

with torch.cuda.amp.autocast():
    out = model(input_X, input_h)