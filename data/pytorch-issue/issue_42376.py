# torch.rand(3)  # Input is a 1D tensor matching the parameter's shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.param = nn.Parameter(torch.tensor([1.0, 2.0, 3.0]))
        self.optimizer = torch.optim.SGD([self.param], lr=0.1, momentum=0.9)
        self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
            self.optimizer, max_lr=6.0, total_steps=10
        )

    def forward(self, x):
        # Simple forward pass to use the parameter with input
        return self.param * x

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 3-element tensor to match the parameter's shape
    return torch.rand(3)

