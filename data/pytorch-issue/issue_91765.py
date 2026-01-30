#!/usr/bin/env python3
import matplotlib.pyplot as plt
import torch

# Setup optimizer + scheduler
epochs = 5
steps_per_epoch = 10
params = torch.randn(10).requires_grad_(True)
optimizer = torch.optim.SGD([params], lr=0.01, momentum=0.9)
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=0.01, steps_per_epoch=steps_per_epoch, epochs=epochs
)

# Collect lrs
lrs = []
for step in range(epochs * steps_per_epoch):
    optimizer.step()
    scheduler.step()
    lrs.append(optimizer.param_groups[0]["lr"])

# Create plot
plt.figure()
plt.title("OneCycleLR Learning Rate Scaling")
plt.plot(lrs)
plt.yscale("log")
plt.xlabel("Steps")
plt.ylabel("Learning Rate")
plt.savefig("/tmp/OneCycleLR.png")

# Print LRs
print("Min lr: ", scheduler.optimizer.param_groups[0]["min_lr"])
print("Actual last lr: ", lrs[-1])
print("Last three lrs: ", lrs[-3:])