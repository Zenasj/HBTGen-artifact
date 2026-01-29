# torch.rand(1, dtype=torch.float)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def forward(self, x):
        s, x_tail = x[..., 0], x[..., 1:]
        outputs = [s]
        for xs_i in torch.unbind(x_tail, dim=-1):
            s = s + xs_i
            outputs.append(s)
        return torch.stack(outputs, dim=-1)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, dtype=torch.float)

