# (torch.tensor([-668044.5], dtype=torch.float32), torch.rand(1, dtype=torch.float32))
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, inputs):
        x, y = inputs
        return F.hinge_embedding_loss(x, y, margin=0.11, reduction='sum')

def my_model_function():
    return MyModel()

def GetInput():
    x = torch.tensor([-668044.5], dtype=torch.float32)
    y = torch.rand(1, dtype=torch.float32)
    return (x, y)

