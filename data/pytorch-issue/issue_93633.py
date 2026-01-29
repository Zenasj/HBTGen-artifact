# torch.rand(B, 2, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, train_X, train_Y):
        super().__init__()
        # Mock components of SingleTaskGP's structure
        self.mean_module = nn.Linear(2, 1)  # Simulates ConstantMean
        self.covar_module = nn.Sequential(
            nn.Linear(2, 10),
            nn.ReLU(),
            nn.Linear(10, 1)
        )  # Simulates ScaleKernel + MaternKernel
        # Store training data as buffers (as in GP models)
        self.register_buffer('train_X', train_X)
        self.register_buffer('train_Y', train_Y)
    
    def forward(self, x):
        # Dummy forward pass mimicking GP behavior
        mean = self.mean_module(x)
        covar = self.covar_module(x)
        return mean, covar  # Return mean and covariance as outputs

def my_model_function():
    # Generate dummy training data for initialization
    train_X = torch.rand(10, 2)
    train_Y = torch.rand(10, 1)
    return MyModel(train_X, train_Y)

def GetInput():
    # Random input matching the expected (B, 2) shape
    return torch.rand(5, 2, dtype=torch.float)

