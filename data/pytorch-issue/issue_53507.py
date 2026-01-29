# torch.rand(0)  # No input required
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, *args, **kwargs):
        mask = torch.zeros(100, dtype=torch.bool)
        indices = (torch.rand(25) * mask.shape[0]).to(torch.int64)
        mask[indices] = True
        return mask

def my_model_function():
    return MyModel()

def GetInput():
    return ()

