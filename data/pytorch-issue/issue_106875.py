# torch.rand(1, dtype=torch.float32)  # Inferred input shape from the issue

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # No additional layers or parameters needed for this simple function

    def forward(self, x):
        with torch.autograd.graph.disable_saved_tensors_hooks("ERROR"):
            y = x + 1
            print("HI")
            return y + 2

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(())

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)
# compiled_model = torch.compile(my_model_function())(GetInput())

