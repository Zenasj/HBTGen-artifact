# torch.rand(4, 1, dtype=torch.float32)
import torch
import torch.nn as nn

class OldMaskGenerator(nn.Module):
    def forward(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class NewMaskGenerator(nn.Module):
    def forward(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1).float()
        mask = mask.masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        self.old_mask = OldMaskGenerator()
        self.new_mask = NewMaskGenerator()

    def forward(self, x):
        sz = x.size(0)
        old_mask = self.old_mask(sz)
        new_mask = self.new_mask(sz)
        return torch.tensor(not torch.allclose(old_mask, new_mask), dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(4, 1)

