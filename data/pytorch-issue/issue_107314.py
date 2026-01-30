import torch

def fn(x):
    return torch.fft(x)

# fn(torch.randn(10, 10))
torch.compile(fn, backend="eager")(torch.randn(10, 10))