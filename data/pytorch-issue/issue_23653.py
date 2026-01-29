# torch.rand(10, dtype=torch.float32)  # Inferred input shape for the model

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # No specific parameters or layers needed for this example

    def forward(self, x):
        # Perform the slicing operation and return the sum
        slize = [1, 2, 3, 4]
        y = x[slize]
        return y.sum()

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(10, requires_grad=True)

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)
# output.backward()  # This will work
# output.backward()  # This will raise an error without retain_graph=True

