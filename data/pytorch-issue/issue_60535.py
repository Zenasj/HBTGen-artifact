# torch.rand(1, 1, 3, 3, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.conv = nn.Conv2d(1, 5, kernel_size=3, bias=False)
        # Set the conv weights to be five 3x3 identity matrices
        self.conv.weight.data = torch.eye(3, 3).unsqueeze(0).repeat(5, 1, 1, 1)

    def forward(self, x):
        return self.conv(x)

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    input_example = torch.zeros(1, 1, 3, 3)
    # Set the middle pixel to be a one (the rest are zeros)
    input_example[:, :, 1, 1] = 1.0
    return input_example

