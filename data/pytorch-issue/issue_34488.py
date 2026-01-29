# torch.rand(1, 10, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.weight_ih = torch.nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        
    def forward(self, x):
        # Using matmul instead of mm to avoid the ONNX export issue
        gates = x.matmul(self.weight_ih.t())
        return gates

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel(10)

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(1, 10, dtype=torch.float32)

