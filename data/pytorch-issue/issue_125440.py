# torch.rand(4, dtype=torch.float32)  # Input tensor: [step, lambd, alpha, lr]
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        step = x[0].item()
        lambd = x[1]
        alpha = x[2]
        lr = x[3]
        
        # Compute eta for both ASGD variants
        term1 = 1 + lambd * lr * step
        eta1 = lr / term1.pow(alpha)  # Single-tensor style: (step) in exponent base
        eta2 = lr / (1 + lambd * lr * (step ** alpha))  # Multi-tensor style: (step^alpha) in denominator
        
        # Return True if they differ beyond 1e-8 tolerance
        return torch.tensor([not torch.allclose(eta1, eta2, atol=1e-8)], dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate test input with higher lambd to expose differences
    step = torch.randint(1, 100, (1,)).float()
    lambd = torch.rand(1) * 0.1  # Up to 0.1 for noticeable differences
    alpha = torch.tensor([0.75])  # Default ASGD alpha
    lr = torch.tensor([0.01])     # Default ASGD learning rate
    return torch.cat([step, lambd, alpha, lr])

