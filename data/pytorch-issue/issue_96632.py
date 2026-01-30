import random

import torch
B=300; N = 640

torch.random.manual_seed(0)
torch.cuda.manual_seed(0)

# Placeholders used for capture
A = torch.randn(B, N, N, device='cuda',requires_grad=True)
b = torch.randn(B, N, 1, device='cuda')
v = torch.ones(B, N, 1, device='cuda')

# Do not run grad related operations on the default stream --- use side stream
# y = torch.bmm(A, b) # this command runs
# dy0dA = torch.autograd.grad(y[:,0,0], A, v[:,0, 0]) # verified that this command runs.

# warmup
s = torch.cuda.Stream()
s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    for i in range(3):
        y = torch.bmm(A, b) # this command runs -- this is an assumed simple forward task
        dy0dA = torch.autograd.grad(y[:,0,0], A, v[:,0,0])
torch.cuda.current_stream().wait_stream(s)

# capture
g = torch.cuda.CUDAGraph()
with torch.cuda.graph(g):
    y = torch.bmm(A, b) # this command runs
    dy0dA = torch.autograd.grad(y[:,0,0], A, v[:,0, 0])

real_inputs = [torch.rand_like(A) for _ in range(10)]
real_targets = [torch.rand_like(b) for _ in range(10)]

for data, target in zip(real_inputs, real_targets):
    # Fills the graph's input memory with new data to compute on
    A = A.detach()
    A.copy_(data)
    b.copy_(target)
    g.replay()
    print(dy0dA)