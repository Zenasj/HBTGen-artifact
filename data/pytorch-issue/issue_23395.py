# torch.rand(2, dtype=torch.float32)  # Inferred input shape from the example

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        y = x / 2
        z = x.sqrt() + 1
        w = torch.where(x == 0., y, z)
        return w

def my_model_function():
    return MyModel()

def GetInput():
    # Return a tensor that matches the input expected by MyModel
    return torch.tensor([16., 0.], requires_grad=True)

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)
# print(output)

