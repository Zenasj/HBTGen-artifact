# torch.rand(B, 3, dtype=torch.float32)
import torch
import torch.nn as nn
import math

class CustomLinear(nn.Linear):
    def reset_parameters(self):
        # Initialize weights as per documentation: uniform from -sqrt(1/in_features) to sqrt(1/in_features)
        nn.init.uniform_(
            self.weight,
            a=-math.sqrt(1.0 / self.in_features),
            b=math.sqrt(1.0 / self.in_features)
        )
        if self.bias is not None:
            nn.init.zeros_(self.bias)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        in_features = 3
        out_features = 5
        self.linear_kaiming = nn.Linear(in_features, out_features)  # Uses default Kaiming initialization
        self.linear_uniform = CustomLinear(in_features, out_features)  # Uses doc-specified uniform initialization
    
    def forward(self, x):
        out_kaiming = self.linear_kaiming(x)
        out_uniform = self.linear_uniform(x)
        max_diff = torch.max(torch.abs(out_kaiming - out_uniform))
        return max_diff < 1e-5  # Returns True if outputs are within 1e-5 tolerance

def my_model_function():
    return MyModel()

def GetInput():
    B = 2  # Batch size (arbitrary choice for testing)
    in_features = 3  # Matches the model's input dimension
    return torch.rand(B, in_features, dtype=torch.float32)

