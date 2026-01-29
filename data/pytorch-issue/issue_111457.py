# torch.rand(2048, 2048, dtype=torch.bfloat16), torch.rand(2048, 512, dtype=torch.bfloat16), torch.rand(512, 2048, dtype=torch.bfloat16)

import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, inputs):
        a, b, d = inputs
        res = a @ b
        res = res @ d
        return res

def my_model_function():
    return MyModel()

def GetInput():
    in_feature = 2048
    output_feature = 512  # Using first element from [512, 2048, 4096]
    a = torch.randn(2048, in_feature, dtype=torch.bfloat16)
    b = torch.randn(in_feature, output_feature, dtype=torch.bfloat16)
    d = torch.randn(output_feature, in_feature, dtype=torch.bfloat16)
    return (a, b, d)

