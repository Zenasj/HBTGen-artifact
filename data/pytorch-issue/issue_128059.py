import torch

@torch.compile(fullgraph=True)
def fn(x):
    with torch.cuda.device(x.device.index):
        x = x + 1

    return x

x = torch.randn(10, device="cuda")
print(fn(x))