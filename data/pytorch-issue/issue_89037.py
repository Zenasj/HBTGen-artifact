from torch._C import DispatchKey
import torch
from functorch.experimental.cond import cond
from torch.fx.experimental.proxy_tensor import make_fx

# A hack to get DispatchKey.PythonDispatcher without updating pybind and rebuilding pytorch.
# Should really be just: DispatchKey.PythonDispatcher
# Placed here for quick repro.
def DispatchKey_PythonDispatcher():
    return torch._C._dispatch_keyset_full_after(DispatchKey.CPU).highestPriorityTypeId()

def true_fn(x):
    return x.sin()


def false_fn(x):
    return x.cos()


def f(x, y):
    return cond(y, true_fn, false_fn, [x])

# without this fallthrough, we will simply miss an impl for this key, which will error during make(tracing_mode="symbolic").
cond.fallthrough(DispatchKey_PythonDispatcher())

graph = make_fx(f, tracing_mode="symbolic")(torch.ones(3, 2), torch.tensor(False))