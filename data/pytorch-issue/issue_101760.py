import torch


params = [
    {"params": [torch.randn(32, 32, device="cuda") for _ in range(100)]},
    {"params": [torch.randn(32, 32, device="cuda") for _ in range(100)]},
]
grads = [
    [torch.randn(32, 32, device="cuda") for _ in range(100)],
    [torch.randn(32, 32, device="cuda") for _ in range(100)],
]
optimizer = torch.optim.Adam(params, fused=True)


for _ in range(100):
    for i, param_groups in enumerate(params):
        for p, g in zip(param_groups["params"], grads[i]):
            p.grad = g
        optimizer.step()
        optimizer.zero_grad()