import torch

param = torch.rand(2, 3, dtype=torch.float32, device='cuda', requires_grad=True)
param_c = param.clone().detach().requires_grad_(True)

def closure():
    param.grad = torch.ones_like(param) * 2
    return param.grad

def closure_c():
    param_c.grad = torch.ones_like(param_c) * 2
    return param_c.grad

optimizer = torch.optim.AdamW([param])
optimizer_c = torch.optim.AdamW([param_c])

def loop(opt, c):
    opt.step(c)

compiled_loop = torch._dynamo.optimize("eager")(loop)

print(f"before compiled loop: {param=}")
compiled_loop(optimizer, closure)
print(f"after compiled loop: {param=}")

print(f"before eager loop: {param_c=}")
loop(optimizer_c, closure_c)
print(f"after eager loop: {param_c=}")