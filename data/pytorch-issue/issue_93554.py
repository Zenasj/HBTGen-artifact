import torch.nn as nn

class Conv_Bn_Relu(nn.Module):
    def __init__(self):
        super(Conv_Bn_Relu, self).__init__()

        self.conv = torch.nn.Conv2d(6, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3, eps=0.001)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    traced_gm = torch.jit.trace(gm.eval(), example_inputs).eval()
    traced_gm = torch.jit.freeze(traced_gm)
    print(traced_gm.graph_for(*example_inputs))
    return traced_gm

model = Conv_Bn_Relu().to(memory_format=torch.channels_last).eval()
x = torch.randn(3, 6, 10, 10).to(memory_format=torch.channels_last)

# In the weight prepack process, aten::conv is replaced by their optimized counterparts in Intel® Extension for PyTorch* via the registration mechanism.
model = ipex.optimize(model)

# jit
with torch.no_grad():
    traced_gm = torch.jit.trace(model, x).eval()
    traced_gm = torch.jit.freeze(traced_gm)
    print(traced_gm.graph_for(x))

# dynamo
m = torchdynamo.optimize(compiler)(model)

with torch.no_grad():
    m(x)

import torch
import intel_extension_for_pytorch as ipex
import torch.nn.functional as F
from typing import List


class Conv_Bn_Relu(torch.nn.Module):
    def __init__(self):
        super(Conv_Bn_Relu, self).__init__()

        self.conv = torch.nn.Conv2d(6, 3, 3)
        self.bn = torch.nn.BatchNorm2d(3, eps=0.001)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)), inplace=True)

def compiler(gm: torch.fx.GraphModule, example_inputs: List[torch.Tensor]):
    traced_gm = torch.jit.trace(gm.eval(), example_inputs).eval()
    traced_gm = torch.jit.freeze(traced_gm)
    print(traced_gm.graph_for(*example_inputs))
    return traced_gm

model = Conv_Bn_Relu().to(memory_format=torch.channels_last).eval()
x = torch.randn(3, 6, 10, 10).to(memory_format=torch.channels_last)

# In the weight prepack process, aten::conv is replaced by their optimized counterparts in Intel® Extension for PyTorch* via the registration mechanism.
model = ipex.optimize(model)

# dynamo
m = torch.compile(model, backend=compiler)

with torch.no_grad():
    m(x)