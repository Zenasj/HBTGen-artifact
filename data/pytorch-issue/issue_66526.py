# torch.rand(B, D), torch.rand(B, D)  # Input is a tuple of two tensors
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b = inputs
        return F.cosine_similarity(a, b)

def my_model_function():
    return MyModel()

def GetInput():
    # Example input dimensions; adjust based on actual data
    B, D = 8, 512
    a = torch.rand(B, D, dtype=torch.float32)
    b = torch.rand(B, D, dtype=torch.float32)
    return (a, b)

