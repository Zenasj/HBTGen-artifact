# torch.rand(10000, 10000, dtype=torch.float64, device='cuda')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        result = torch.matmul(x, x)
        torch.cuda.synchronize()  # Explicit synchronization to trigger error
        return result

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10000, 10000, dtype=torch.float64, device='cuda')

