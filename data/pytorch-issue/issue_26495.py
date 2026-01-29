# torch.rand(2, 3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        empty_out = torch.empty(0, dtype=x.dtype, device=x.device)
        success1 = 0
        success0 = 0
        try:
            _ = torch.pow(x, 1, out=empty_out.clone())
            success1 = 1
        except RuntimeError:
            pass
        try:
            _ = torch.pow(x, 0, out=empty_out.clone())
            success0 = 1
        except RuntimeError:
            pass
        return torch.tensor([success1, success0], dtype=torch.long)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(2, 3)

