import torch
import torch.nn as nn

class M(torch.nn.Module):
    def forward(self, x):
        return x * 3.2
        
ep = torch.export._trace._export(M(), (x,), strict=False, pre_dispatch=True)

class GraphModule(torch.nn.Module):
        def forward(self, arg0_1: "f32[1, 3, 299, 299]"):
            # No stacktrace found for following nodes
            mul: "f32[1, 3, 299, 299]" = torch.ops.aten.mul.Tensor(arg0_1, 3.2);  arg0_1 = None
            return (mul,)