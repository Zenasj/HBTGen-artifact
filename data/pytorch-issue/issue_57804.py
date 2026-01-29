# torch.rand(1, dtype=torch.float32)
import torch
from torch import nn

def _to_tensor(val):
    if isinstance(val, State):
        return val.state
    return val

class State:
    def __init__(self):
        super().__init__()
        self.state = torch.ones(1)
    def __repr__(self):
        return f"{self.__class__.__name__}: {self.state}"
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        if kwargs is None:
            kwargs = {}
        with torch._C.DisableTorchFunction():
            args = [_to_tensor(arg) for arg in args]
            kwargs = {key: _to_tensor(val) for key, val in kwargs.items()}
            ret = func(*args, **kwargs)
            return ret

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.state = State()
        self.param = nn.Parameter(torch.randn(3))
    
    def forward(self, x):
        # Intentionally triggers the order-dependent error with broadcast_tensors
        return torch.broadcast_tensors(self.param, self.state)[0]

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(1)

