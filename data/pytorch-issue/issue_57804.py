import torch


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


state=State()
torch.broadcast_tensors(state, state) # works
torch.broadcast_tensors(state, torch.randn(3)) # works
torch.broadcast_tensors(state, Parameter(torch.randn(3))) # works
torch.broadcast_tensors(Parameter(torch.randn(3)), state) # TypeError