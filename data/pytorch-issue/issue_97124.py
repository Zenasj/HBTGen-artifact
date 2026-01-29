# torch.rand(1, 2, dtype=torch.int32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        x = x + x
        x = x.to(dtype=torch.float32)
        return x * 0.8

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.zeros([1, 2], dtype=torch.int32)

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)
# print(output)

