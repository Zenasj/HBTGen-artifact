# torch.rand(144, 144, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        # Perform argmax and argmin operations
        argmax_result = torch.argmax(x, dim=1)
        argmin_result = torch.argmin(x, dim=1)
        return argmax_result, argmin_result

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.randn(144, 144, dtype=torch.float32)

