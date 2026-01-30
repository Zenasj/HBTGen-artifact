import torch
import os

os.environ["MTL_CAPTURE_ENABLED"]="1"

foo_c = torch.compile(torch.ops.aten.lgamma)

x = torch.arange(256, device="mps", dtype=torch.float16).sin()
with torch.mps.profiler.metal_capture("lgamma"):
    y = foo_c(x)

print(y)