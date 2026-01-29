# torch.rand(3, 3, dtype=torch.int)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.eager_foo = lambda x: torch.empty_like(x, dtype=None)
        try:
            self.scripted_foo = torch.jit.script(lambda x: torch.empty_like(x, dtype=None))
            self.has_scripted = True
        except Exception:
            self.has_scripted = False

    def forward(self, x):
        eager_out = self.eager_foo(x)
        if self.has_scripted:
            jit_out = self.scripted_foo(x)
            return torch.allclose(eager_out, jit_out)
        else:
            return torch.tensor(False, dtype=torch.bool)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.empty(3, 3, dtype=torch.int)

