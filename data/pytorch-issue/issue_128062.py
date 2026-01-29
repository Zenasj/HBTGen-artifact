# torch.rand(B, C, H, W, dtype=...)  # In this case, the input shape is (B, C) where B is batch size and C is the number of features
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.linear = nn.Linear(3, 3, bias=False)

    def forward(self, x):
        return self.linear(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand(2, 3)

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)

