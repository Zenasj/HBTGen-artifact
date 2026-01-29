# torch.rand(1, 1, 4, 4, dtype=torch.float32)  # Add a comment line at the top with the inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.pool = nn.MaxPool2d(2, stride=2, return_indices=True)
        self.unpool = nn.MaxUnpool2d(2, stride=2)

    def forward(self, x):
        output, indices = self.pool(x)
        unpooled44 = self.unpool(output, indices)
        unpooled55 = self.unpool(output, indices, output_size=torch.Size([1, 1, 5, 5]))
        return unpooled44, unpooled55

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.tensor([[[[1.0, 2.0, 3.0, 4.0],
                           [5.0, 6.0, 7.0, 8.0],
                           [9.0, 10.0, 11.0, 12.0],
                           [13.0, 14.0, 15.0, 16.0]]]])

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# unpooled44, unpooled55 = model(input_tensor)

