import torch
torch.cuda.set_sync_debug_mode(1)
x=torch.randn(10, device="cuda")
x.nonzero()
y=torch.randn((), device="cuda")

if y:
    print("something")
torch.multinomial(x.abs(), 10, replacement=False)
torch.randperm(20000, device="cuda")
ind = torch.randint(10, (3,), device="cuda")
mask = torch.randint(2, (10,), device="cuda", dtype=torch.bool)
val = torch.randn((), device="cuda")
x[mask]=1.
x[mask] = val
torch.cuda.synchronize()