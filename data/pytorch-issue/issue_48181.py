# torch.rand(3) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(3, 4)
        self.norm_layer = nn.utils.spectral_norm(self.linear)

    def forward(self, x):
        return self.norm_layer(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(3)

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)

