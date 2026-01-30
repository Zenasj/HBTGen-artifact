import torch

params = [
    torch.rand(2, 3, dtype=torch.float64, device='cuda:0', requires_grad=True),
    torch.rand(2, 3, dtype=torch.float32, device='cuda:0', requires_grad=True),
    torch.rand(2, 3, dtype=torch.float16, device='cuda:0', requires_grad=True),
    torch.rand(2, 3, dtype=torch.bfloat16, device='cuda:0', requires_grad=True),
    torch.rand(2, 3, dtype=torch.float64, device='cuda:1', requires_grad=True),
    torch.rand(2, 3, dtype=torch.float32, device='cuda:1', requires_grad=True),
    torch.rand(2, 3, dtype=torch.float16, device='cuda:1', requires_grad=True),
    torch.rand(2, 3, dtype=torch.bfloat16, device='cuda:1', requires_grad=True),
]

optimizer = torch.optim.AdamW(params)
for p in params:
    p.grad = torch.rand_like(p)

def loop():
    optimizer.step()

compiled_loop = torch._dynamo.optimize("eager")(loop)
compiled_loop()