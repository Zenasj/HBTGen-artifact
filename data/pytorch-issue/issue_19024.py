# torch.rand(1, 1, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.discriminator1 = nn.Linear(1, 1, bias=True).cuda()
        self.discriminator2 = nn.DataParallel(nn.Linear(1, 1, bias=True)).cuda()
        # Synchronize initial weights between discriminators
        self.discriminator2.module.load_state_dict(self.discriminator1.state_dict())
    
    def forward(self, x):
        # Compute outputs from both discriminators and return their difference
        out1 = self.discriminator1(x)
        out2 = self.discriminator2(x)
        return out1 - out2  # Difference tensor indicates discrepancy

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 1, dtype=torch.float32, device='cuda')

