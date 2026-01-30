import torch
from torch.utils._python_dispatch import TorchDispatchMode,_get_current_dispatch_mode

class EnableParitalMode(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        mode = _get_current_dispatch_mode()
        print("in dispatch" ,mode) # None
        func(*args, **kwargs)
    

b = torch.Tensor(1,4,8)
with EnableParitalMode():
    mode = _get_current_dispatch_mode()
    print("in outer ", mode) # EnableParitalMode
    a = torch.ones_like(b)