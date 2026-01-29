# torch.rand(3, dtype=torch.float32)
import torch
import torch.nn as nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('input_tensor', torch.tensor([4., 5., 6.], dtype=torch.float32))
        self.register_buffer('append_tensor', torch.tensor([1., 2., 3.], dtype=torch.float32))
        self.n = 5
        self.dim = 0

    def forward(self, x):
        return torch.diff(
            input=self.input_tensor,
            dim=self.dim,
            n=self.n,
            prepend=x,
            append=self.append_tensor
        )

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(3, dtype=torch.float32)

