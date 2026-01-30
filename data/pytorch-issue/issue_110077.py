import torch
import torch._dynamo

def fn(x):
  return x.byte()

x = torch.tensor(-2, dtype=torch.float32)

raw = fn(x)
print("raw", raw)

run = torch._dynamo.optimize()(fn)
opt = run(x)
print("opt", opt)
assert raw == opt

from torch._inductor import config
config.debug = True