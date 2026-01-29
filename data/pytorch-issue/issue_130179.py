# torch.rand(B, 3, 1, 1, dtype=torch.float32)
import torch
from torch import nn

@torch.library.custom_op("mylib::foo", mutates_args={"self_"})
def foo(self_: torch.Tensor) -> None:
    self_.sin_()

class MyModel(nn.Module):
    def forward(self, x):
        foo(x)
        return x

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1, 3, 1, 1, dtype=torch.float32)

