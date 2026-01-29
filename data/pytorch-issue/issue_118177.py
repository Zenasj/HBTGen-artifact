# torch.rand(8, 6, 2, 6, 6, 1, 1, 4, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, input):     
        return torch.argmax(input=torch.fliplr(input=input))

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(8, 6, 2, 6, 6, 1, 1, 4, 1, dtype=torch.float32)

