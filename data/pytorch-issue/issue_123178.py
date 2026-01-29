# torch.rand(1, dtype=torch.float32)  # Inferred input shape for scalar tensor

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, activation_func):
        super(MyModel, self).__init__()
        self.activation_func = activation_func

    def forward(self, x):
        y = self.activation_func(x).sum()
        return y

def my_model_function():
    # Return an instance of MyModel with a specific activation function
    # Here we use F.leaky_relu as an example
    return MyModel(F.leaky_relu)

def GetInput():
    # Return a random scalar tensor input that matches the input expected by MyModel
    return torch.tensor(3.0, requires_grad=True)

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)
# output.backward()
# print(input_tensor.grad)

