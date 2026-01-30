import torch
import torch.nn as nn
import torch.nn.functional as F

class OneConv(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3)

    def forward(self, x):
        x = self.conv1(x)
        return x

model=OneConv()
input = torch.zeros((1,1,16,16),dtype=torch.float)
scriptmodule = torch.jit.trace(model,input)
scriptmodule.save("export.pt")