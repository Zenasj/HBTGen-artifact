import torch

with torch.autograd.profiler.profiler() as prof:
    with torch.autograd.profiler.record_function("foo"):
        rpc.rpc_async(...)