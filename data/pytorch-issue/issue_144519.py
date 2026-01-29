# torch.rand(128, 128, dtype=torch.float16), torch.rand(128, 4096, dtype=torch.float16), torch.rand(4096, 128, dtype=torch.float16)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        b, x, y = inputs
        return torch.addmm(b, x, y)

def my_model_function():
    return MyModel()

def GetInput():
    M, N, K = 128, 128, 4096
    dtype = torch.float16
    b = torch.randn(M, N, dtype=dtype).cuda()
    x = torch.randn(M, K, dtype=dtype).cuda()
    y = torch.randn(K, N, dtype=dtype).cuda()
    return (b, x, y)

