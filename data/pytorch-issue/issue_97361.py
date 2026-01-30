import torch

def pt2_optimizer_step(optimizer):
    @torch.compile()
    def f():
        optimizer.step()
    f()

params = [torch.rand(20005, 768, dtype=torch.float32, device='cpu') for _ in range(194)]
for p in params:
    p.grad = torch.rand_like(p)

o = torch.optim.AdamW(params)
pt2_optimizer_step(o)

def pt2_optimizer_step(optimizer):
    @torch.compile()
    def f():
        optimizer.step()
    f()

params = [torch.rand(77, 512, dtype=torch.float32, device='cuda') for _ in range(1354)]
for p in params:
    p.grad = torch.rand_like(p)

o = torch.optim.AdamW(params, foreach=False)
pt2_optimizer_step(o)

def pt2_optimizer_step(optimizer):
    @torch.compile()
    def f():
        optimizer.step()
    f()

params = [torch.rand(64, 3, 7, 7, dtype=torch.float32, device='cuda') for _ in range(364)]
for p in params:
    p.grad = torch.rand_like(p)

o = torch.optim.ASGD(params, foreach=False)
pt2_optimizer_step(o)