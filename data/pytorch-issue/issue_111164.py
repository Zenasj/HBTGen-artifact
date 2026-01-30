import torch
import torch._inductor.config

torch._inductor.config.triton.inject_relu_bug_TESTING_ONLY = "runtime_error"

def fn(x, y):
    return (x @ y).relu()

x, y = [torch.rand((16, 16), device='cuda') for _ in range (2)]
torch.compile(fn)(x, y)