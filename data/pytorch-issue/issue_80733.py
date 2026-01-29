# torch.rand(B, 3, dtype=torch.float32, device='cuda:0')
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Linear(3, 1)
    
    def forward(self, x):
        return self.layer(x)

def my_model_function():
    model = MyModel()
    model.to(torch.device('cuda:0'))
    return model

def GetInput():
    return torch.rand(1, 3, dtype=torch.float32, device='cuda:0')

