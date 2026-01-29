# torch.rand(4, 1, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
    
    def generate_old_mask(self, S):
        mask = torch.triu(torch.ones(S, S), diagonal=0).to(torch.bool)
        old_mask = torch.zeros(S, S, dtype=torch.float)
        old_mask[~mask] = float('-inf')
        return old_mask
    
    def generate_new_mask(self, S):
        mask = torch.triu(torch.ones(S, S), diagonal=1).to(torch.bool)
        new_mask = torch.zeros(S, S, dtype=torch.float)
        new_mask[mask] = float('-inf')
        return new_mask
    
    def forward(self, x):
        S = x.size(0)
        old_mask = self.generate_old_mask(S)
        new_mask = self.generate_new_mask(S)
        return (old_mask != new_mask).any().float()

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 1, 1, dtype=torch.float32)

