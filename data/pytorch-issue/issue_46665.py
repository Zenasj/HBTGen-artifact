# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn
import contextlib

@contextlib.contextmanager
def do_nothing():
    yield

class MyModel(nn.Module):
    @do_nothing()
    def forward(self, x):
        return x

def my_model_function():
    return MyModel()

def GetInput():
    B, C, H, W = 1, 3, 224, 224
    return torch.rand(B, C, H, W, dtype=torch.float32)

