# torch.rand(10, 10, dtype=torch.float32, device="cuda")
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        # This should create a copy but due to the bug, it shares memory
        y = torch.asarray(x, copy=True, device="cuda")
        # Modify the copied tensor - this will also modify the original due to the bug
        y[0, 0] = 10
        return y

def my_model_function():
    return MyModel()

def GetInput():
    # Create initial tensor on CUDA
    return torch.empty(10, 10, dtype=torch.float32, device="cuda")

