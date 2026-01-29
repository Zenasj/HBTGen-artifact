# torch.rand(1)  # Dummy input tensor, not used in computation
import torch
import numpy as np

class MyModel(torch.nn.Module):
    def forward(self, x):
        a = np.random.uniform(low=-1, high=1, size=(20, 1))
        return torch.tensor([a, a, a, a], dtype=torch.float64, device="cpu")

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)  # Dummy input matching required signature

