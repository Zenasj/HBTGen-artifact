# torch.randn(B, 1, 50257, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.softmax = nn.Softmax(dim=-1)
    
    def forward(self, x):
        return self.softmax(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1024, 1, 50257, dtype=torch.float32).to('cuda')

