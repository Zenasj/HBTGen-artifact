import torch

def wrapper_fn(x):
    with torch.autograd.graph.disable_saved_tensors_hooks("ERROR"):
        y = x + 1
        print("HI")
        return y + 2

x = torch.randn(())

a = wrapper_fn(x)
opt = torch.compile(wrapper_fn, backend='eager', fullgraph=False)
e = opt(x)