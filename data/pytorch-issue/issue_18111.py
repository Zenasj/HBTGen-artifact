# torch.rand(2, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Convert dense input to sparse tensor, which triggers derivative issue if requires_grad=True
        sparse_x = x.to_sparse()
        # Compute loss based on sparse tensor values to require gradient computation
        return sparse_x.values().sum()

def my_model_function():
    return MyModel()

def GetInput():
    # Create a 2x3 tensor with requires_grad=True to trigger backward pass through to_sparse()
    return torch.randn(2, 3, dtype=torch.float32, requires_grad=True)

