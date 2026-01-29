import torch
import torch.nn as nn

# Mock Float8Linear class as a placeholder (actual implementation may vary)
class Float8Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True, **kwargs):
        super().__init__(in_features, out_features, bias)
        # Placeholder for float8-specific configurations (e.g., cast_configs)
        # These parameters are inferred from the issue's model structure
        self.cast_configs = kwargs.get("cast_configs", "i:dyn_ten,w:dyn_ten,go:dyn_ten")

    def forward(self, input):
        # Mock forward pass with float8 conversion logic (simplified)
        return super().forward(input)

# torch.rand(B, C=784, dtype=torch.float32) ← Input shape inferred from Linear(784, 1024)
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(784, 1024, bias=True),  # First layer (non-float8)
            Float8Linear(1024, 1024, bias=True, cast_configs="i:dyn_ten,w:dyn_ten,go:dyn_ten"),  # Second layer (float8)
            Float8Linear(1024, 4096, bias=True, cast_configs="i:dyn_ten,w:dyn_ten,go:dyn_ten")  # Third layer (float8, different shape)
        )
    
    def forward(self, x):
        return self.encoder(x)

def my_model_function():
    # Returns the model instance with default initialization
    return MyModel()

def GetInput():
    # Generates input matching the model's expected shape (B, 784)
    batch_size = 8  # Matches the repro's input size (8, 4096 in later layers)
    return torch.rand(batch_size, 784, dtype=torch.float32)  # Input dtype inferred from issue context

