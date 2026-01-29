# torch.rand(3, 3, 1, dtype=torch.float16)
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.bn = nn.BatchNorm1d(3, eps=1e-3, momentum=0.1)
        
    def forward(self, x):
        out = self.bn(x)
        out = F.relu(out)
        return out

def my_model_function():
    model = MyModel()
    model.eval()
    model.half()  # Match the user's half() conversion
    return model

def GetInput():
    return torch.randn(3, 3, 1, dtype=torch.float16) * 10  # Matches user's input scaling

