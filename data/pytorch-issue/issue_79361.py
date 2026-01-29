# torch.rand(1, 1, 4, 4, dtype=torch.float32) ← Add a comment line at the top with the inferred input shape
import torch
import numpy as np
from onnx.backend.test.case.node import resize

class MyModel(torch.nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def forward(self, x):
        return torch.nn.functional.interpolate(x, size=[1, 1], mode='bilinear')

def my_model_function():
    # Return an instance of MyModel, include any required initialization or weights
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    return torch.rand([1, 1, 4, 4], dtype=torch.float32)

