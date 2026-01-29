# torch.rand(2, 3, dtype=torch.float32)
import torch
from torch import nn

class OldPath(nn.Module):
    def forward(self, t):
        return torch.autograd.Variable(t.clone().detach(), requires_grad=True)

class NewPath(nn.Module):
    def forward(self, t):
        return torch.autograd.Variable(t, requires_grad=True)

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.old = OldPath()
        self.new = NewPath()
    
    def forward(self, input_tensor):
        t = input_tensor.clone()  # Original tensor
        old_v = self.old(t)
        new_v = self.new(t)
        
        # Perform in-place modification on the original tensor
        t.add_(1)  # Triggers version counter difference
        
        # Compute outputs
        y_old = old_v * old_v
        y_new = new_v * new_v
        
        # Check gradients and error handling
        old_ok = True
        new_ok = True
        try:
            y_old.sum().backward()
        except:
            old_ok = False
        
        try:
            y_new.sum().backward()
        except RuntimeError:
            new_ok = False
        
        # Return True if outcomes differ (new throws error, old doesn't)
        return torch.tensor(old_ok != new_ok, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 3, dtype=torch.float32)

