import torch

params = [
    torch.rand(2, 3, dtype=torch.float64, device='cpu', requires_grad=True),
]

optimizer = torch.optim.SGD(params, lr=0.02, momentum=0.9)
for p in params:
    p.grad = torch.empty_like(p)

def loop():
    optimizer.step()

compiled_loop = torch._dynamo.optimize("eager")(loop)
compiled_loop()