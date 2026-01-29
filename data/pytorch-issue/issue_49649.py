# torch.rand(B, 3, 224, 224, dtype=torch.float32)  # Example input shape for image-like data
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.hardsigmoid = nn.Hardsigmoid()  # Core component from the issue context

    def forward(self, x):
        return self.hardsigmoid(x)

def my_model_function():
    return MyModel()  # Directly return the model instance

def GetInput():
    # Generate a random input tensor matching expected model input shape
    return torch.rand(1, 3, 224, 224, dtype=torch.float32)

