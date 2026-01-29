# torch.rand(500, 10000, dtype=torch.float32, device='cuda')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Linear(10000, 10000)  # Matches the TestModel structure from the issue

    def forward(self, x):
        return self.layer(x)

def my_model_function():
    # Initialize model on CUDA as in the original code
    model = MyModel()
    model.to(torch.device('cuda'))
    return model

def GetInput():
    # Generate input matching the model's expected dimensions and device
    return torch.randn(500, 10000, dtype=torch.float32, device='cuda')

