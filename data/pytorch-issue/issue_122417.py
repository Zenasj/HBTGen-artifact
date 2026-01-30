import torch
from torch.utils._python_dispatch import TorchDispatchMode
from torch._dispatch.python import enable_python_dispatcher

class Mode(TorchDispatchMode):
    def __torch_dispatch__(self, *args, **kwargs):
        return None

with enable_python_dispatcher(), Mode():
    while True:
        torch.ones(3, 4)