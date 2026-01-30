import torch.nn as nn

import torch
import functorch
from functorch.compile import aot_function, aot_module, make_boxed_func
from typing import List
import torch._dynamo
from torch._dynamo.backends.common import aot_autograd

device = torch.device('cuda')

class test_model(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def forward(self, x):
        return torch.pow(x,3)

model = test_model()
model.eval()

x = torch.rand([5], device=device)
x = x.requires_grad_(True)

torch._dynamo.config.verbose = True
torch._dynamo.reset()

graph_modules = []

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
  graph_modules.append(gm)
  return make_boxed_func(gm.forward)

model_aot = aot_module(model, fw_compiler=my_compiler, bw_compiler=my_compiler)
result_aot = model_aot(x)
result_backward_aot = torch.autograd.grad(result_aot.sum(),x,retain_graph=True)[0]

forward_graphmodule = graph_modules[0]
for node in forward_graphmodule.graph.nodes:
    if node.stack_trace:
        print(node.stack_trace)

backward_graphmodule = graph_modules[1]
for node in backward_graphmodule.graph.nodes:
    if node.stack_trace:
        print(node.stack_trace)

graph_modules = []

def my_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
  graph_modules.append(gm)
  return make_boxed_func(gm.forward)

### AOT Module ###
model_aot = aot_module(model, fw_compiler=my_compiler, bw_compiler=my_compiler)
result_aot = model_aot(arg0, arg_1, kw_arg0=kw_arg0, kw_arg1=kw_arg1)

### AOT Module Simplified ###
aot_simplified_args = [arg0, arg1, kw_arg0, kw_arg1]
model_aot = aot_module_simplified(model, aot_simplified_args, fw_compiler=my_compiler, bw_compiler=my_compiler)
result_aot = model_aot(*aot_simplified_args)

### torch.compile with 'aot_eager' backend ###
from torch._dynamo.backends.debugging import boxed_nop

def aot_eager_compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    graph_modules.append(gm)
    return boxed_nop(gm, example_inputs)

aot_eager_compiler = aot_autograd(fw_compiler=aot_eager_compiler, bw_compiler=aot_eager_compiler)

model_aot = torch.compile(model, backend=aot_eager_compiler)
result_aot = model_aot(arg0, arg_1, kw_arg0=kw_arg0, kw_arg1=kw_arg1)

### torch.compile with a custom backend wrapped by aot_autograd ###
my_compiler = aot_autograd(fw_compiler=my_compiler, bw_compiler=my_compiler)

model_aot = torch.compile(model, backend=my_compiler)
result_aot = model_aot(arg0, arg_1, kw_arg0=kw_arg0, kw_arg1=kw_arg1)