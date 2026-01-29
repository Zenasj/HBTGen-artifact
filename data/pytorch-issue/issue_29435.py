# torch.rand(62, 59, dtype=torch.float32)  # Inferred input shape from the issue

import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        qx = torch.quantize_per_tensor(x, 1.0, 0, torch.qint32)
        qy = qx.permute([1, 0])
        return qy.dequantize()

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(62, 59)

# Example usage:
# model = my_model_function()
# input_tensor = GetInput()
# output = model(input_tensor)

