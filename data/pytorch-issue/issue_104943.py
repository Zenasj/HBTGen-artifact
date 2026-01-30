import torch.nn as nn

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
        # Define the layers in the model
        self.fc = nn.Linear(in_features=16, out_features=16, bias=True)
        self.elu1 = nn.ELU(alpha=1, inplace=True)
        self.elu2 = nn.ELU(alpha=1, inplace=True)

    def forward(self, x):
        # Define the forward pass
        x = self.fc(x)
        x = self.elu1(x)
        x = self.elu2(x)
        
        return x

md = MyModel()
ip = torch.rand([16,16])
md(ip) # No error 
torch.compile(md)(ip) # RuntimeError : one of the variables needed for gradient computation has been modified by an inplace operation: [torch.FloatTensor [16, 16]]