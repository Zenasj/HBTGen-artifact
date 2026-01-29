# torch.rand(51, 51, 51, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Permute input to (Z, Y, X) for tmp0 and (Y, Z, X) for tmp1 as per kernel logic
        tmp0 = x.permute(2, 1, 0)  # Z,Y,X
        tmp1 = x.permute(1, 2, 0)  # Y,Z,X
        output = 3 * tmp0 + tmp1  # 3*tmp0 + tmp1 from kernel's computation
        return output

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(51, 51, 51, dtype=torch.float32)

