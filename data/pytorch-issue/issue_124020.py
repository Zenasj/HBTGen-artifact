# torch.rand((), dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p0 = nn.Parameter(torch.randn(()), requires_grad=False)

    def forward(self, v0_0):
        # In-place sigmoid on p0 (modifies the parameter)
        v6_0 = torch.sigmoid_(self.p0)
        # In-place atan2_ with swapped parameter order (as in original issue)
        v5_0 = torch.Tensor.atan2_(v0_0, other=self.p0)
        return v6_0, v5_0

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn((), dtype=torch.float32)

