# torch.rand(3000, 200), torch.randn((),).expand(200)  # matrix and vector inputs
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        matrix, vector = inputs
        out_tensor = torch.full((vector.size(0),), 0., device=matrix.device)
        r0 = torch.mv(matrix, vector)
        r1 = torch.mv(matrix, vector, out=out_tensor)
        mask = torch.logical_or(torch.isnan(r0), torch.isnan(r1))
        r0_masked = r0.masked_fill(mask, 0.)
        r1_masked = r1.masked_fill(mask, 0.)
        return torch.all(r0_masked == r1_masked)

def my_model_function():
    return MyModel()

def GetInput():
    n = 3000
    m = 200
    matrix = torch.randn(n, m)
    vector = torch.randn((),).expand(m)
    return (matrix, vector)

