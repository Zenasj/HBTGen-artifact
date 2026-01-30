import torch

def bad_backend(gm, example_inputs):
    raise RuntimeError("bad")

@torch.compile(backend=bad_backend)
def fn(x):
    return x + 1

import torch._dynamo
# torch._dynamo.config.verbose = True
fn(torch.randn(3, 3))