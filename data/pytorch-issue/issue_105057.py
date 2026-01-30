import torch.nn as nn

import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.fc = nn.Linear(in_features=16, out_features=16)
        self.avg = nn.AdaptiveAvgPool1d(output_size=[0])

    def forward(self, x):
        x = self.fc(x)
        print('fc', x.size())
        x = self.avg(x)
        
        return x

md = MyModel()
ip = torch.rand([16,16,16])
md(ip) ## No Error
torch.compile(md)(ip)