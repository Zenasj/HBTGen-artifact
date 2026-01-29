# torch.rand(B, C, H, W, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.l = torch.nn.LPPool2d(2, 3)
        self.n = torch.nn.LPPool2d(2, (7, 1))  # Triggers scripting error for tuple kernel_size

    def forward(self, x):
        o = []
        o.append(self.l(x))
        o.append(self.n(x))  # Triggers scripting error for tuple kernel_size
        o.append(torch.nn.functional.lp_pool2d(x, float(2), 3))
        o.append(torch.nn.functional.lp_pool2d(x, 2, 3))  # Triggers scripting error for int norm_type
        o.append(torch.nn.functional.lp_pool2d(x, float(2), (7, 1)))  # Triggers kernel_size tuple error
        return o

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 7, 7, dtype=torch.float32)

