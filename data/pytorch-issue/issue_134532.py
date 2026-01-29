# torch.rand(B, M, K, dtype=torch.float16)  # Input shape and dtype for prologue fusion example
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        # Example dimensions: K=8 (input features), N=5 (output features)
        self.linear = nn.Linear(8, 5)  # Matches input's last dimension (K)
    
    def forward(self, x):
        # Simulate prologue fusion candidate: dtype conversion (dequantization)
        x = x.float()  # This op could be fused into the matmul's prologue
        return self.linear(x)

def my_model_function():
    return MyModel()

def GetInput():
    # Example dimensions: B=2 (batch), M=4 (sequence), K=8 (features)
    B, M, K = 2, 4, 8
    return torch.rand(B, M, K, dtype=torch.float16)  # Matches the input comment's dtype

