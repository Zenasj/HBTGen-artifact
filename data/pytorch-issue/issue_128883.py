# torch.rand(2, 2, dtype=torch.bfloat16), torch.rand(2, 2, dtype=torch.bfloat16)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        x, y = inputs
        # Apply aten.view.dtype to reinterpret data type without copying
        x = torch.ops.aten.view.dtype(x, torch.float16)
        y = torch.ops.aten.view.dtype(y, torch.float16) + 1
        return torch.matmul(x, y)

def my_model_function():
    return MyModel()

def GetInput():
    # Generate two tensors of shape (2,2) with bfloat16 dtype as required
    x = torch.randn(2, 2, dtype=torch.bfloat16)
    y = torch.randn(2, 2, dtype=torch.bfloat16)
    return (x, y)

