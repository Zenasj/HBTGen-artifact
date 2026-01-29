# torch.rand(B, C, dtype=torch.float32, device='cuda')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(1500, 1500, bias=False)
    
    def forward(self, x):
        return self.linear(x)

def my_model_function():
    model = MyModel()
    model.to('cuda')  # Match device from original issue's example
    return model

def GetInput():
    return torch.rand(1, 1500, dtype=torch.float32, device='cuda')

