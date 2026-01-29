# torch.rand(B, C, H, W, dtype=torch.float32, requires_grad=True)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        tmp = x * 2  # Replicates the temporary tensor scenario from the issue example
        return tmp  # Returns the intermediate tensor to enable gradient computation against non-leaf nodes

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a 4D tensor with requires_grad=True to match the example's leaf tensor
    return torch.rand(1, 3, 32, 32, dtype=torch.float32, requires_grad=True)

