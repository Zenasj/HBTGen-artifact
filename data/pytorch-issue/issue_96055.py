from typing import List
import torch
from torch import Tensor
from functorch.compile import aot_function
from torch._subclasses.fake_tensor import FakeTensorMode
from torch._prims.context import TorchRefsMode

def func(scale: Tensor,shape):
    return torch.reshape(scale,shape)


def custom_backend(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    print("custom backend called with FX graph:")
    gm.graph.print_tabular()
    return gm.forward

scale=torch.ones(3)


with FakeTensorMode(allow_non_fake_inputs=True):
    scale=scale.clone()

with TorchRefsMode(strict=True):
    func = aot_function(func,fw_compiler=custom_backend)
    print(func(scale,shape=(-1,)))

import torch
from torch.library import Library

from torch.fx.experimental.proxy_tensor import (
    proxy_call,
    get_innermost_proxy_mode,
    disable_proxy_modes_tracing
)


def leaf(library: Library, schema: Optional[str]=None):
    def wrapper(func):
        def wrapped(*args, **kwargs):
            proxy_mode = get_innermost_proxy_mode()
            if not proxy_mode or getattr(wrapped,'recursive',None):
                with disable_proxy_modes_tracing():
                    return func(*args, **kwargs)
            wrapped.recursive = True
            res =  proxy_call(
                proxy_mode,
                getattr(getattr(torch.ops,library.ns), func.__name__).default,
                args,
                kwargs,
            )
            wrapped.recursive = False
            return res
        library.define(schema)
        library.impl(func.__name__, wrapped,"Autograd")
        return getattr(getattr(torch.ops,library.ns), func.__name__).default
    return wrapper

mylib = Library("mylib", "DEF")
@leaf(mylib,"foo(Tensor x)->Tensor")
def foo(x):
    return x*2

# In general, we don't want to make modules leaves. In principle, users of
    # this tracer might want to override this in order to turn a couple specific
    # modules into leaves in the traced graph.