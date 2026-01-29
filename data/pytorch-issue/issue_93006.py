# torch.rand(160, 256, 3, 1024, dtype=torch.float32)  # Inferred input shape

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        # Define a simple model that uses topk
        self.topk_k = 32
        self.dim = -1

    def forward(self, x):
        # Perform the topk operation
        values, indices = x.topk(self.topk_k, dim=self.dim)
        return values, indices

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(160, 256, 3, 1024, dtype=torch.float32).to("cuda")

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# values, indices = model(input_tensor)

