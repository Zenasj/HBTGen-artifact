# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.param = nn.Parameter(torch.randn(2, 2, requires_grad=True))
        self.optimizer = torch.optim.SGD([self.param], 0.1)
        self.scheduler1 = torch.optim.lr_scheduler.ConstantLR(self.optimizer, factor=0.1, total_iters=2)
        self.scheduler2 = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.9)
        self.scheduler = torch.optim.lr_scheduler.ChainedScheduler([self.scheduler1, self.scheduler2])

    def forward(self, x):
        return x  # Dummy forward pass; model parameters are used in optimizer setup

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 2, 2, 2, dtype=torch.float32)

