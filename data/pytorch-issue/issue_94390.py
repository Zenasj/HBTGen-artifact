# torch.rand(1, 10, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, z):
        output = z.clone()
        interpolation = torch.zeros(z.size(1), device=z.device)
        for i in range(z.size(1)):
            output[:, i] = interpolation[i]
        return output

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 10)

