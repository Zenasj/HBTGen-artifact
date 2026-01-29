# torch.rand(6, dtype=torch.float32)
import torch
from torch import nn

class ModelA(nn.Module):
    def __init__(self):
        super().__init__()
        mask_tensor = torch.tensor([True, False], dtype=torch.bool)
        self.register_buffer('mask', mask_tensor)
        
    def forward(self, x):
        x.view(3, 2).masked_fill_(self.mask.unsqueeze(0), torch.finfo(x.dtype).max)
        return x

class ModelB(nn.Module):
    def __init__(self):
        super().__init__()
        mask_tensor = torch.tensor([True, False], dtype=torch.bool)
        self.register_buffer('mask', mask_tensor)
        
    def forward(self, x):
        x = x.view(3, 2)
        x.masked_fill_(self.mask.unsqueeze(0), torch.finfo(x.dtype).max)
        return x.view(-1)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.model_a = ModelA()
        self.model_b = ModelB()
        
    def forward(self, x):
        x_a = x.clone()
        x_b = x.clone()
        out_a = self.model_a(x_a)
        out_b = self.model_b(x_b)
        return torch.tensor(torch.allclose(out_a, out_b), dtype=torch.bool).unsqueeze(0)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(6, dtype=torch.float32)

