# torch.rand(4, 6, 4, 1, dtype=torch.float32)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, input):
        v1 = F.softmax(input, 1)
        v2 = v1.transpose(0, 3)
        return v2.div_(2.0)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand([4, 6, 4, 1])

