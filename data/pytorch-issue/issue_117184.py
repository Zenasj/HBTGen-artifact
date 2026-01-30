import torch
param = torch.rand(2, 3, dtype=torch.float32, device='cuda')  # moving this inside f would work! 

def f(p):
    p.grad = torch.rand_like(p)
    p.grad = p.grad.to_sparse()  # why does dynamo think p.grad at this moment is empty?

compiled_f = torch._dynamo.optimize("eager")(f)
compiled_f(param)