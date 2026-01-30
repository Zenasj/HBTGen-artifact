import torch

a = torch.rand(4, 2, device="cuda")

with torch.cuda.stream(second_stream):
    torch.mul(a, 5, out=a)