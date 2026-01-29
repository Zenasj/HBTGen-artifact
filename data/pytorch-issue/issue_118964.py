# torch.rand(7,3, dtype=torch.float32), torch.rand(7,3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.p0 = torch.nn.Parameter(torch.rand(7, 3, dtype=torch.float32), requires_grad=False)

    def forward(self, inputs):
        i1, i2 = inputs
        r1, r2 = torch._C._linalg.linalg_qr(A=i1)
        without_this_no_error = torch.pow(exponent=i2, input=self.p0, out=i1)
        return r1, r2

def my_model_function():
    return MyModel()

def GetInput():
    return (
        torch.rand(7, 3, dtype=torch.float32),
        torch.rand(7, 3, dtype=torch.float32)
    )

