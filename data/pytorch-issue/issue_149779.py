# torch.rand(B, 10), torch.randint(0, 10, (B,))  # Input shapes for probs and labels
import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, inputs):
        probs, labels = inputs
        return F.nll_loss(probs, labels)

def my_model_function():
    return MyModel()

def GetInput():
    B = 32  # Matched to test script's initial batch size
    probs = torch.rand(B, 10, device='cuda')
    labels = torch.randint(0, 10, (B,), device='cuda')
    return probs, labels

