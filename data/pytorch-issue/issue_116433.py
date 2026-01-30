import torch

def f(x):
    tmp = x + 1
    return tmp.view(-1, 1, 2)

x = torch.arange(8, device='cuda', dtype=torch.float32)
out = f(x)
compiled_out = torch.compile(f)(x)
print(out.shape)  # []
print(compiled_out.shape)
print(out.stride())
print(compiled_out.stride())

# prints:
torch.Size([4, 1, 2])
torch.Size([4, 1, 2])
(2, 2, 1)
(2, 0, 1)