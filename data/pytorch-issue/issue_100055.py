import torch
import torch.nn as nn
import torch._dynamo as dynamo
from torch.fx.experimental.proxy_tensor import make_fx
from torch._dispatch.python import enable_python_dispatcher
from torch._guards import detect_fake_mode

def compiler(gm, example_inputs):
    fake_mode = detect_fake_mode(example_inputs)
    fake_inputs = [fake_mode.from_tensor(i) if isinstance(i, torch.Tensor) else i
                   for i in example_inputs]
    with fake_mode, enable_python_dispatcher():
        fx_graph = make_fx(gm, pre_autograd=True)(*fake_inputs)
        print(fx_graph.graph)
    return gm.forward


@dynamo.optimize(compiler, dynamic=True)
def f(x, w, b):
    z = torch.nn.functional.linear(x, w, b)
    return z


w = torch.randn(20, 10)
b = torch.randn(20)
f(torch.randn(1, 2, 10), w, b)
f(torch.randn(1, 3, 10), w, b)

fake_mode = FakeTensorMode()

w = torch.randn(20, 10)
b = torch.randn(20)
i1 = torch.randn(1, 2, 10)

s = ShapeEnv()
fake_mode = FakeTensorMode(shape_env=s)
fake_inputs = [fake_mode.from_tensor(i) if isinstance(i, torch.Tensor) else i
               for i in [i1, w, b]]
with fake_mode, enable_python_dispatcher():
    g1 = make_fx(f, pre_autograd=True)(*fake_inputs)