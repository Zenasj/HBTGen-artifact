# torch.rand(B, 10, dtype=torch.float32)  # Inferred input shape: (batch_size, 10)

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.use_sigmoid = False  # Flag to switch between Model and Model2 behavior

    def forward(self, x):
        x = self.relu(x)
        if self.use_sigmoid:
            x = self.sigmoid(x)
        return x

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    device = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.randn(8, 10, device=device)

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)

