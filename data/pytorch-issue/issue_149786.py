py
import torch

def func(a):
    return torch.softmax(a, dim=-1, dtype=torch.float32)

a = torch.randn(4, 16, dtype=torch.float16, device="cuda")


g = torch.cuda.CUDAGraph()

torch.cuda.synchronize()
with torch.cuda.graph(g):
    out = func(a)

torch.cuda.synchronize()
g.replay()
torch.cuda.synchronize()

print(out.shape)