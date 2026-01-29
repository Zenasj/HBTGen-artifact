# torch.rand(2, 1, dtype=torch.complex64)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        res = torch.exp(x)
        # Return boolean tensor indicating if first element's imaginary part is NaN (bug presence)
        return torch.isnan(res[0, 0].imag).view(1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.tensor([[float("inf")+0.0000j], [-1.-8.7423e-08j]], dtype=torch.complex64)

