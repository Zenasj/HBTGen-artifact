import torch
shifts = torch.arange(
    0, 32, 8,
    dtype=torch.int64,
    device="cuda")
tensor = torch.randint(
    2,
    size=(16, 1),
    dtype=torch.int64,
    device="cuda")

def func(a, b, shape):
    c = (a >> b)
    return c.view(shape)

# Pass
print(func(tensor, shifts, (16, 4)).shape)
# TorchRuntimeError: Failed running call_method view( (FakeTensor(..., device='cuda:0', size=(16, 1), dtype=torch.int64), (16, 4)), **{}): shape '[16, 4]' is invalid for input of size 16
print(torch.compile(func)(tensor, shifts, (16, 4)).shape)