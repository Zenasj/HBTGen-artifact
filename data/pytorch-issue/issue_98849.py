import torch.nn as nn

import torch
from torch import nn
class D(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(16, 16)
        self.d = nn.Dropout(p=0.5)
    def forward(self, x):
        x = self.fc(x)
        x = self.d(x)
        return x

model = D()
model = torch.compile(model, dynamic=True)
tensor = torch.randn(5, 10, 16)
model.train()
model(tensor)