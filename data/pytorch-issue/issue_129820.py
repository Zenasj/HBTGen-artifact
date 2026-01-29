# torch.rand(7,7), torch.rand(()), torch.rand(7,7)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, inputs):
        v3_0, v1_0, v0_0 = inputs
        v4_0 = torch.lt(v1_0, other=40, out=v3_0)
        print(v1_0.shape, "==>", v4_0.shape)
        return v4_0

def my_model_function():
    return MyModel()

def GetInput():
    v3_0 = torch.rand(7,7)
    v1_0 = torch.rand(())
    v0_0 = torch.rand(7,7)
    return (v3_0, v1_0, v0_0)

