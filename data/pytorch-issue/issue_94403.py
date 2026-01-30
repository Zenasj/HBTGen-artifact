import torch
from torch.utils._python_dispatch import TorchDispatchMode, _pop_mode_temporarily
import torch._dispatch.python
from functools import partial

class DummyDispatcher(TorchDispatchMode):
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        return func(*args, **kwargs)

x = torch.zeros([10, 10], device="cuda")
x.record_stream(torch.cuda.Stream())
with DummyDispatcher():
    # Boom
    x.record_stream(torch.cuda.Stream())

class TorchFuncMockNoDispatch:
    """
    Wraps a method to call it without the custom
    pytorch dispatcher
    """
    def __init__(self, pt_impl):
        self.pt_impl = pt_impl
    def __get__(self, obj, c):
        return partial(self, obj)
    def __call__(self, obj, *args, **kwargs):
        with _pop_mode_temporarily():
            return self.pt_impl(obj, *args, **kwargs)

class DispatcherWithoutBrokenFuncs(TorchDispatchMode):
    TENSOR_FUNCS_NO_DISPATCH = [
        # Can't convert Stream argument to Python object
        'record_stream'
    ]
    def __enter__(self) -> None:
        self._pt_impls = {}
        for k in self.TENSOR_FUNCS_NO_DISPATCH:
            impl = getattr(torch.Tensor, k)
            self._pt_impls[k] = impl
            setattr(torch.Tensor, k, TorchFuncMockNoDispatch(impl))
        return super().__enter__()
    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        for k in self.TENSOR_FUNCS_NO_DISPATCH:
            setattr(torch.Tensor, k, self._pt_impls[k])
        return super().__exit__(exc_type, exc_val, exc_tb)