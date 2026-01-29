import torch
import torch.nn as nn

# torch.rand(B, C, dtype=torch.float32)  # Input shape inferred from original issue's (1, 3) example
class MyModel(nn.Module):
    def forward(self, x):
        return x.repeat_interleave(2)  # Original problematic implementation without specifying 'dim'

def my_model_function():
    return MyModel()  # Returns the model instance causing ONNX export failure

def GetInput():
    return torch.rand(1, 3)  # Matches the original input shape (1, 3) from the issue's minimal repro

