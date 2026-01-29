# torch.rand(1, 5, 10, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv1 = nn.Conv1d(5, 10, 4)
        self.conv2 = nn.Conv1d(10, 5, 3)

    def forward(self, x):
        y = self.conv1(x)
        y = self.conv2(y)
        return y

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn((1, 5, 10)).cuda()  # Ensure the input is on the same device as the model

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)

