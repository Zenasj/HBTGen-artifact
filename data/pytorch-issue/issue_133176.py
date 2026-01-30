import torch
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.passes.reinplace import _FunctionalizationMetadataProp
from torch.fx.experimental.symbolic_shapes import ShapeEnv
from torch._dynamo.source import ConstantSource
from torch.fx.experimental.sym_node import SymNode


def fn(x, index):
    res = torch.select(x, 0, index)
    return res.relu()


a = torch.randn((4, 8, 16, 16), requires_grad=False)
graph_module = make_fx(fn)(a, 2)
print(graph_module.print_readable(False))

val = 2
shape_env = ShapeEnv()
symbol = shape_env.create_symbol(val, source=ConstantSource(
    f"__testing_only{len(shape_env.var_to_val)}"))
sym_int = torch.SymInt(SymNode(symbol, shape_env, int, hint=val))
example_inputs = [a, sym_int]
_FunctionalizationMetadataProp(graph_module).propagate(*(example_inputs))

print("done")