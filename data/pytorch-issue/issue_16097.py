import torch
import torch.nn as nn

class InnerModule(nn.Module):
    
    def forward(self, x, y_dict):
        return x + y_dict['y']
     
class OuterModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.inner = InnerModule()
        
    def forward(self, x, y):
        y_dict = dict(y=y)
        return self.inner(x, y_dict)
    
self = OuterModule()
x, y = torch.randn(3), torch.randn(3)

torch.jit.trace(self, (x, y)) # fails

class AlternativeInnerModule(nn.Module):
    
    def forward(self, x, y_tuple):
        return x + y_tuple[0]

y_dict = dict(y=y)
class InnerModule(nn.Module):
    
    def forward(self, x, y_dict):
        return x + y_dict['y']
     
class OuterModule(nn.Module):
    
    def __init__(self):
        super().__init__()
        self.inner = InnerModule()
        
    def forward(self, x, y):
        y_dict = dict(y=y)
        return self.inner(x, y_dict)