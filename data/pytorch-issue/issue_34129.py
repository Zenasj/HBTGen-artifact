# torch.rand(H, W, dtype=torch.float32)
import torch
from torch import nn

def tril_onnx(inputs: torch.FloatTensor, diagonal: int = 0) -> torch.FloatTensor:
    arange = torch.arange(inputs.size(0), device=inputs.device)
    arange2 = torch.arange(inputs.size(1), device=inputs.device)
    mask = arange.unsqueeze(-1).expand(-1, inputs.size(1)) >= (arange2 - diagonal)
    return inputs.masked_fill(mask == 0, 0)

class MyModel(nn.Module):
    def forward(self, x):
        return tril_onnx(x)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(10, 10, dtype=torch.float32)

