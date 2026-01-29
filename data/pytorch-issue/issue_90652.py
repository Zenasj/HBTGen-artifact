# torch.rand(1, dtype=torch.float32)
import torch
from torch.utils._mode_utils import no_dispatch

class MyModel(torch.nn.Module):
    def forward(self, x):
        with torch._subclasses.FakeTensorMode():
            a = torch.rand([100])
            with torch._subclasses.CrossRefFakeMode():
                with no_dispatch():
                    b = torch.zeros_like(a)
        return b

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)  # Dummy input (not used in forward)

