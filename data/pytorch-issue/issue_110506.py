import torch.nn as nn

import time
import torch
from torch.optim import Adam, SGD

def compile_opt(opt_compiled):
    torch._dynamo.eval_frame.TorchPatcher.patch()

    step_fn = opt_compiled.step.__wrapped__
    def fn():
        step_fn(opt_compiled)

    return torch.compile(fn, backend="inductor", fullgraph=True)

optim_cls = SGD
NUM_PARAMS = 1000
kwargs = { "lr": 0.01, "foreach": True }

torch._dynamo.reset()
# torch._inductor.metrics.reset()
input = torch.ones([10, 10], device="cuda:0")
model = torch.nn.Sequential(
    *[torch.nn.Linear(10, 10, device="cuda:0") for _ in range(NUM_PARAMS)]
)

input = torch.ones([10, 10], device="cuda:0")
model(input).sum().backward()
opt_compiled = optim_cls(model.parameters(), **kwargs)
compiled_step = compile_opt(opt_compiled)

with torch.set_grad_enabled(False):
    start_time = time.time()
    compiled_step()
    print("compile opt took: %s seconds", time.time() - start_time)