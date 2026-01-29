# torch.rand(1, dtype=torch.complex128)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, z):
        # Method 1: 0.5 * (z - z.conj())
        out1 = 0.5 * (z - z.conj())
        
        # Method 2: z.imag
        out2 = z.imag
        
        return out1, out2

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.ones(1, requires_grad=True, dtype=torch.complex128)

# Example usage:
# model = my_model_function()
# z = GetInput()
# out1, out2 = model(z)
# out1.backward()
# out2.backward()
# print(z.grad)

