import torch.nn as nn

import torch
from torch.testing._internal.common_utils import run_tests, TemporaryFileName, TestCase
from torch.utils import ThroughputBenchmark

from contextlib import nullcontext

from torch._dynamo import config
from torch._inductor import config as inductor_config


class TwoLayerNet(torch.jit.ScriptModule):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(2 * H, D_out)

    @torch.jit.script_method
    def forward(self, x1, x2):
        h1_relu = self.linear1(x1).clamp(min=0)
        h2_relu = self.linear1(x2).clamp(min=0)
        cat = torch.cat((h1_relu, h2_relu), 1)
        y_pred = self.linear2(cat)
        return y_pred


class TwoLayerNetModule(torch.nn.Module):
    def __init__(self, D_in, H, D_out):
        super().__init__()
        self.linear1 = torch.nn.Linear(D_in, H)
        self.linear2 = torch.nn.Linear(2 * H, D_out)

    def forward(self, x1, x2):
        h1_relu = self.linear1(x1).clamp(min=0)
        h2_relu = self.linear1(x2).clamp(min=0)
        cat = torch.cat((h1_relu, h2_relu), 1)
        y_pred = self.linear2(cat)
        return y_pred

Module = TwoLayerNetModule
dtype = torch.bfloat16
config.error_on_recompile = True
inductor_config.cpp_wrapper = True
inductor_config.freezing = True
D_in = 10
H = 5
D_out = 15
B = 8

autocast = dtype != torch.float32
module = Module(D_in, H, D_out)

input = (torch.randn(B, D_in), torch.randn(B, D_in))

with torch.no_grad(), torch.amp.autocast("cpu", enabled=autocast, dtype=dtype):
    torch._dynamo.reset()
    module(*input)
    module = torch.compile(module)
    module(*input)
    module(*input)

with torch.autograd.profiler.profile() as prof:
    with torch.no_grad(), torch.amp.autocast("cpu", enabled=autocast, dtype=dtype):
        module(*input)
print(prof.key_averages().table(sort_by="self_cpu_time_total"))