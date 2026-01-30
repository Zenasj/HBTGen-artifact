import torch
import torch.nn as nn

from torch import nn
from torch import optim
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt

model = nn.Linear(1, 1)
max_lr = 1.0
momentum = 0.9
epochs = 150
batches_per_epoch = 1000
optimiser = optim.SGD(model.parameters(), lr=max_lr, momentum=momentum)
scheduler = OneCycleLR(optimiser, max_lr, epochs=epochs, steps_per_epoch=batches_per_epoch,
                       pct_start=0.475, anneal_strategy='linear', cycle_momentum=True,
                       base_momentum=0.85, max_momentum=momentum, div_factor=10,
                       final_div_factor=1e4)

ys = []
for _ in range(epochs):
    for _ in range(batches_per_epoch):
        ys.append(optimiser.param_groups[0]['lr'])
        scheduler.step()
plt.title('OneCycleLR schedule')
plt.ylabel('Learning rate')
plt.xlabel('Step')
plt.plot(ys, c='red')
plt.show()