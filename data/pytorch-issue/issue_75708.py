import torch
import torch.nn as nn

# torch.rand(12, 64, 4096, dtype=torch.float).cuda()  # Main input tensor t1's shape
class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.i1 = 1  # Fixed integer parameter
        self.i2 = 1  # Fixed integer parameter

    def forward(self, inputs):
        t1, t2, t3, t4 = inputs
        v1 = torch.sub(t3, t4, alpha=self.i2)
        v2 = torch.mul(v1, t2)
        v3 = torch.reshape(t1, [1, 12, 64, 4096])
        v4 = torch.add(v3, v2, alpha=self.i1)
        v5 = torch._softmax(v4, -1, False)
        v6 = torch.reshape(v5, [12, 64, 4096])
        return v6

def my_model_function():
    return MyModel()

def GetInput():
    # Generate tensors with exact shapes/dtypes from the issue's repro code
    t1 = torch.rand(12, 64, 4096, dtype=torch.float).cuda()
    t2 = torch.rand((), dtype=torch.double).cuda()
    t3 = torch.rand((), dtype=torch.double).cuda()
    t4 = torch.rand(1, 1, 1, 4096, dtype=torch.float).cuda()
    return (t1, t2, t3, t4)

