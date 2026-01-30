import torch

class T(torch.Tensor):
    def __new__(cls, elem):
        return torch.Tensor._make_wrapper_subclass(cls, elem.shape, dtype=elem.dtype)

    def __init__(self, elem):
        self.elem = elem

    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func is torch.ops.aten.detach.default:
            (self,) = args
            return cls(torch.ops.aten.detach.default(self.elem))
        raise NotImplementedError(f"{func}")
    
x = torch.randn(3)
y = T(torch.randn(3))
    
print("ok")
with torch.inference_mode():
    x.detach()
    
print("fail")
with torch.inference_mode():
    y.detach()

with torch.inference_mode():
    x = torch.tensor(1.)
T(x) # should this be an inference tensor?