# torch.rand(B, C, H, W, dtype=...)  # In this case, the input shape is (3, 3) for a 2D tensor
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, x):
        # Calculate the Frobenius norm of the input tensor
        return x.norm(dim=(0, 1))

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(3, 3, device='cuda')

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)

