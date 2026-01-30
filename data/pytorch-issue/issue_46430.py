import torch.nn as nn

import torch
import torch.nn.quantized as nnq
from torch.fx.symbolic_trace import Tracer
from torch.fx import GraphModule

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ff = nnq.FloatFunctional()
        
    def forward(self, x):
        x = self.ff.add_relu(x, x)
        return x
    
class CustomTracer(Tracer):
    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name : str) -> bool:
        print('type:', type(m))
        return (m.__module__.startswith('torch.nn') and not isinstance(m, torch.nn.Sequential)) or \
            type(m) == nnq.FloatFunctional()


m = M()
m = GraphModule(m, CustomTracer().trace(m))
print(m)
print('type of float functional:', type(m.ff))