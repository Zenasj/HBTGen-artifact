# torch.rand(2, dtype=torch.float32, device='cuda')  # Inferred input shape
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.s = torch.cuda.Stream()  # Stream as model attribute
    
    def forward(self, t):
        tmp1 = torch.mul(t, 5)
        tmp2 = torch.add(tmp1, 2)
        with torch.cuda.stream(self.s):
            r = torch.relu(tmp2)
        s1 = torch.add(r, 2)
        s2 = torch.cos(s1)
        return s2

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, dtype=torch.float32, device='cuda')

