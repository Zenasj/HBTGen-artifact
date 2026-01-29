# torch.rand(B, 2, 3, dtype=torch.float32)  # Inferred input shape: (batch_size, 2, 3)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # Transpose the tensor along dimensions 1 and 2
        xt = x.transpose(1, 2)
        return xt

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Generate a random tensor input that matches the input expected by MyModel
    # The input shape is (batch_size, 2, 3) with batch_size being dynamic
    batch_size = 1  # Example batch size, can be any positive integer
    return torch.randn(batch_size, 2, 3, dtype=torch.float32)

