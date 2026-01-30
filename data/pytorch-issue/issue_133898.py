import torch

torch._logging.set_logs(recompiles_verbose=True)

param = torch.rand(2, 3, dtype=torch.float, device="cuda", requires_grad=True)
param.grad = torch.rand_like(param)

lr = torch.tensor(0.001, device="cuda")
total_steps = 10000
optimizer = torch.optim.AdamW([param], lr=lr)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=lr, total_steps=total_steps
)

@torch.compile()
def step():
    optimizer.step()
    scheduler.step()


for _ in range(total_steps):
    step()