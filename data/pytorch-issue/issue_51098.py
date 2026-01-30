import torch

@torch.jit.script
def gn(d: torch.device):
    print(d)

print(gn("test"))

opencl:0