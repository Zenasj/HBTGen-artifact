# torch.rand(100, 100, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, B):
        # Compute C[x,y] = sin(B[2x, 2y]) / B[x,y] for x,y in [0,50)
        C = torch.sin(B[::2, ::2]) / B[:50, :50]
        return C

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(100, 100, dtype=torch.float32)

