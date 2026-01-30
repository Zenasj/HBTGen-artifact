import torch

x = torch.empty(10, pinned_memory=True)
y = torch.randn(10, device='cuda')
torch.cuda._sleep(...)
event = torch.cuda.Event().record()
x.copy_(y, non_blocking=True)
# check that event hasn't occurred (i.e that the copy is really non-blocking)
# synchronize
# check that x and y are equal