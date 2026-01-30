import torch
from torch.fx.experimental.proxy_tensor import make_fx
from functorch.experimental import functionalize

def fn(a):
    a = a * 2
    a.relu_()
    return a

input = torch.randn([1, 1])
graph_module = torch.fx.symbolic_trace(fn)
fx_graph = make_fx(functionalize(graph_module))(input)