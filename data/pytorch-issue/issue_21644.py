import torch

a = torch.tensor([1,2,3], dtype=torch.float32, device='cuda')

with torch.cuda.profiler.profile():
    with torch.autograd.profiler.emit_nvtx():
        b = a.sum()