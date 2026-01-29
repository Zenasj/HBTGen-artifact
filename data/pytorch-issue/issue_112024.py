# torch.rand(3, dtype=torch.float32)
import torch
from torch import nn

class TOriginal(torch.Tensor):
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

class TFixed(torch.Tensor):
    def __new__(cls, elem):
        return torch.Tensor._make_wrapper_subclass(cls, elem.shape, dtype=elem.dtype)
    
    def __init__(self, elem):
        self.elem = elem
    
    @classmethod
    def __torch_dispatch__(cls, func, types, args, kwargs=None):
        if func is torch.ops.aten.detach.default:
            (self,) = args
            with torch.inference_mode(False):
                return cls(torch.ops.aten.detach.default(self.elem))
        raise NotImplementedError(f"{func}")

class MyModel(nn.Module):
    def forward(self, x):
        orig_ok = 0
        fixed_ok = 0
        try:
            with torch.inference_mode():
                t_orig = TOriginal(x)
                detached_orig = t_orig.detach()
            orig_ok = 1
        except RuntimeError:
            pass
        
        try:
            with torch.inference_mode():
                t_fixed = TFixed(x)
                detached_fixed = t_fixed.detach()
            fixed_ok = 1
        except RuntimeError:
            pass
        
        # Return 1.0 if fixed works and original fails (as expected)
        return torch.tensor(1.0 if (fixed_ok == 1 and orig_ok == 0) else 0.0, dtype=torch.float32)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randn(3, dtype=torch.float32)

