# Input is a tuple (x, y) with x.shape (B, 1) and y.shape (B)
import torch
from torch import nn

class LossNoSqueeze(nn.Module):
    def forward(self, x, y):
        return torch.mean((x - y) ** 2)

class LossWithSqueeze(nn.Module):
    def forward(self, x, y):
        return torch.mean((x.squeeze() - y) ** 2)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_no_squeeze = LossNoSqueeze()
        self.loss_with_squeeze = LossWithSqueeze()

    def forward(self, inputs):
        x, y = inputs
        loss1 = self.loss_no_squeeze(x, y)
        loss2 = self.loss_with_squeeze(x, y)
        return torch.abs(loss1 - loss2) > 0.01

def my_model_function():
    return MyModel()

def GetInput():
    B = 10  # Inferred from original example's 10 elements
    x = torch.rand(B, 1)
    y = torch.rand(B)
    return (x, y)

