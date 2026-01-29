# torch.rand(2, dtype=torch.float32)
import torch
import numpy as np

class MyModel(torch.nn.Module):
    def forward(self, x):
        return x

def my_model_function():
    return MyModel()

def GetInput():
    # Generate input that avoids negative strides by copying the numpy array
    arr = np.flip(np.array([1, 34]))
    return torch.from_numpy(arr.copy())

