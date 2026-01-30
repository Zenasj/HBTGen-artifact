import torch.nn as nn

# bench_gil
import torch
from torch._inductor import config as inductor_config
inductor_config.cpp_wrapper = True

class SimpleM(torch.nn.Module):
    def __init__(self):
        super(SimpleM, self).__init__()
        self.linear1 = torch.nn.Linear(100, 100)
        self.linear2 = torch.nn.Linear(100, 100)


    def forward(self, x, y):
        return self.linear1(x) + self.linear1(y)

from torch.utils import ThroughputBenchmark
model = torch.compile(SimpleM().bfloat16())
x1 = torch.randn(100, 100).bfloat16()
x2 = torch.randn(100, 100).bfloat16()
with torch.no_grad():
    y = model(x1, x2)
    y = model(x1, x2)

bench = ThroughputBenchmark(model)
bench.add_input(x1, x2)
with torch.no_grad():
    stats = bench.benchmark(
        num_calling_threads=24,
        num_warmup_iters=100,
        num_iters=2400,
    )
print(stats)