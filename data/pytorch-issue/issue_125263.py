# torch.rand(100, dtype=torch.float32)  # Inferred input shape from the issue's repro code
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = nn.Linear(100, 100)  # Matches the Linear layer in the original model
        self.relu = nn.ReLU()  # Added based on forward() reference in logs

    def forward(self, x):
        return self.relu(self.l(x))  # Matches the forward logic from the issue's code

def my_model_function():
    # Returns initialized model on CUDA (as in the original issue)
    return MyModel().to("cuda:0")

def GetInput():
    # Returns input tensor matching (100,) shape from logs, on CUDA device
    return torch.randn(100, dtype=torch.float32, device="cuda:0")

