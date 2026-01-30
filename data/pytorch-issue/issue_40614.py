import torch

torch.jit.script
def remote_fn(t: int):
    return t

torch.jit.script
def local_fn():
    for _ in range(1_000_000):
        fut = rpc.rpc_async("rhs", remote_fn, (42,))
        fut.wait()