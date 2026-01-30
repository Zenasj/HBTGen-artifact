import torch.nn as nn

torch.save(model, model_pytorch_filepath)

torch.__version__
'1.4.0'

import torch
from torch import nn

class Test(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Linear(500, 10)
    def forward(self, x):
        return self.fc(x)

def testing():
    x = torch.randn(4, 500)
    m = Test()
    y = m(x)
    torch.save(m, './test.pth')
testing()