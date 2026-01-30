import torch.nn as nn

import torch
import matplotlib.pyplot as plt

model = torch.nn.Linear(2, 1)
optimizer = torch.optim.SGD(model.parameters(), lr=0.5)
lr_scheduler_1 = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=20, eta_min=0.1)
lr_scheduler_2 = torch.optim.lr_scheduler.CyclicLR(optimizer, base_lr=0.1, max_lr=0.3, step_size_up=1, step_size_down=3)

lrs = []

for i in range(40):
    if i <= lr_scheduler_1.T_max:
        lr_scheduler_1.step()
    else:
        lr_scheduler_2.step()
    lrs.append(
        optimizer.param_groups[0]["lr"]
    )

plt.plot(lrs)