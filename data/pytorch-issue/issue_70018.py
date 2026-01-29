# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.log_softmax = nn.LogSoftmax(dim=1)  # Apply along channel dimension
    
    def forward(self, x):
        return self.log_softmax(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input matching (B, C, H, W) from test case (3,3,100,100)
    B, C, H, W = 3, 3, 100, 100
    return torch.rand(B, C, H, W, dtype=torch.float32)

