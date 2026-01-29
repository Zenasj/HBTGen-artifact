# torch.rand(B, C, dtype=torch.float32, device='cuda')
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(100, 10)  # Matches the Linear layer in the issue's example
        
    def forward(self, x):
        return F.relu(self.lin(x))  # Uses ReLU as in the original model

def my_model_function():
    # Return an instance of MyModel, initialized in eval mode and moved to CUDA
    model = MyModel().eval().cuda()
    return model

def GetInput():
    # Returns a random input tensor matching the model's expected input (CUDA device)
    return torch.randn(8, 100, dtype=torch.float32, device='cuda')

