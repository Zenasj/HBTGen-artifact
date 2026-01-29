# torch.rand(B, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Simulate quantum circuit's expectation value using PyTorch operations
        phi = x[:, :2]  # First two elements for phi
        theta = x[:, 2]  # Third element for theta
        # Dummy expectation value calculation (replaces PennyLane's circuit4)
        exp_val = torch.cos(phi[:,0] + theta) * torch.sin(phi[:,1])
        cost = torch.abs(exp_val - 0.5)**2
        return cost

def my_model_function():
    return MyModel()

def GetInput():
    # Input combines phi (size 2) and theta (size 1) into a single tensor
    return torch.rand(1, 3, dtype=torch.float32, requires_grad=True)

