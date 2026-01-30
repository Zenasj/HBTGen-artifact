py
import torch

def fn(x):
    s = torch.cuda.Stream()
    x = torch.mul(x, 5)
    x = torch.add(x, 2)

    print("foo")

    tcs = torch.cuda.stream(s)
    current_stream = torch.cuda.current_stream()
    s.wait_stream(current_stream)

    with tcs:
        x = torch.relu(x)

    current_stream.wait_stream(s)
    x = torch.add(x, 1)
    x = torch.cos(x)
    return x

x = torch.randn((2, 2), device="cuda")
ref = fn(x)
opt_fn = torch.compile(fn, backend="eager")
res = opt_fn(x)