import torch
import triton

@triton.jit
def kernel(X):
    return

def forward(x):
    kernel[(1, )](x, num_ctas=1)
    return x

torch.compile(forward)(torch.randn(1))