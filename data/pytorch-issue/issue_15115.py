# torch.rand(1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.mu = nn.Parameter(torch.ones(1, requires_grad=True))

    def forward(self, x):
        loss = 0
        for _ in range(3):
            x.detach_()
            new_calc = torch.exp(self.mu)
            x.copy_(new_calc)
            # Using problematic operation that triggers the error
            loss += (x * 2).sum()  
        return loss

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1)

