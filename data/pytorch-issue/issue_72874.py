import torch

optimizer = torch.optim.SGD([torch.tensor(0.5)], lr=0.1)
print(optimizer.param_groups[0]["lr"])  # 0.1, as expected

schedulers = [
    torch.optim.lr_scheduler.ConstantLR(optimizer, factor=1),
    torch.optim.lr_scheduler.ConstantLR(optimizer, factor=0.1)
]
scheduler = torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers, milestones=[10])

print(optimizer.param_groups[0]["lr"])  # 0.01, which is incorrect. It should be 0.1