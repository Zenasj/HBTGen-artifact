# Input: (torch.rand(3, 10, dtype=torch.float32), torch.randint(10, (3,), dtype=torch.int64))
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        loss_fct = torch.nn.CrossEntropyLoss(label_smoothing=0.1)
        return loss_fct(x, y)

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.rand(3, 10)
    y = torch.randint(10, (3,))
    return (x, y)

