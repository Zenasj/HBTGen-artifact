import torch

def func(x):
    # return torch.add(x, 1).numpy()
    return torch.Tensor.numpy(torch.add(x, 1))

x = torch.randn(32, 3, 64, 64)
compiled_func = torch.compile(func)

with torch.no_grad():
    compiled_func(x)