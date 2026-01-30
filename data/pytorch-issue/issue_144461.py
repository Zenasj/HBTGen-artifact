import torch.nn as nn

import torch

class CatDense(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(128, 128)
        
    def forward(self, x) -> torch.Tensor:
        y = self.linear(x)
        return y

class Model(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.linear = torch.nn.Linear(13, 128)
        self.catdense = CatDense()

    def forward(self, dense):
        out = self.linear(dense)
        out = self.catdense(out)
        return out

dtype = torch.float32 # this issue also exists with float16
bs = 256
from torch._inductor import config as inductor_config
from torch._dynamo import config
config.error_on_recompile = True
inductor_config.cpp_wrapper = True
inductor_config.cpp.enable_kernel_profile = True
inductor_config.freezing = True
model = Model()
dense = torch.zeros(bs, 13)
model(dense)

autocast = dtype != torch.float32
with torch.no_grad(), torch.cpu.amp.autocast(enabled=autocast, dtype=dtype):
    print('[Info] Running torch.compile() with default backend')
    model(dense)
    model = torch.compile(model)
    model(dense)
    model(dense)

from torch.utils import ThroughputBenchmark
import contextlib
ctx = contextlib.suppress()
if dtype == 'fp16':
    ctx = torch.cpu.amp.autocast(enabled=autocast, dtype=dtype)
with torch.no_grad(), ctx:
    bench = ThroughputBenchmark(model)
    bench.add_input(dense)
    stats = bench.benchmark(
        num_calling_threads=1,
        num_warmup_iters=200,
        num_iters=300,
    )