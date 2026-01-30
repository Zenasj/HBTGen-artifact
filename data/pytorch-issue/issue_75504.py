import os

import torch
from torch.profiler import ProfilerActivity, profile

os.environ["CUDA_LAUNCH_BLOCKING"] = "1"


def add_one(in_: torch.Tensor):
    return in_ + 1


sample_arg = torch.zeros(10, device="cuda").requires_grad_(True)
add_one_graphed = torch.cuda.graphs.make_graphed_callables(add_one, sample_args=(sample_arg,))

zeros = torch.zeros(10, device="cuda")
out = add_one_graphed(zeros)
assert out[0] == 1

# This works
with profile(activities=[ProfilerActivity.CPU]):
    add_one_graphed(zeros)

# RuntimeError: CUDA error: an illegal memory access was encountered
with profile(activities=[ProfilerActivity.CUDA]):
    add_one_graphed(zeros)