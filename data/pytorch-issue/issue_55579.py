# torch.randint(0, 4, (B,), dtype=torch.long, device='cuda')  # B is batch size
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self, num_classes=4):
        super(MyModel, self).__init__()
        self.num_classes = num_classes

    def forward(self, x):
        return F.one_hot(x, self.num_classes)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 4, (3,), dtype=torch.long, device='cuda')

