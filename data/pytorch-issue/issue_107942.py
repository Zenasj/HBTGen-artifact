# torch.rand(1, 2)  # Input shape is (1, 2)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.layer = nn.Dropout(p=1.0, inplace=False)
    
    def forward(self, inputs):
        return self.layer(inputs)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(1, 2)

