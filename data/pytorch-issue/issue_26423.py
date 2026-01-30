import torch
from torch.nn import Parameter
from torch.optim import SGD
from torch.optim.lr_scheduler import ExponentialLR, StepLR

model = [Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 0.1)

scheduler1 = ExponentialLR(optimizer, gamma=0.9)
scheduler2 = StepLR(optimizer, step_size=5, gamma=0.1)

for epoch in range(10):

    print(epoch, scheduler2.get_last_lr()[0])

    optimizer.step()
    scheduler1.step()
    scheduler2.step()

# Setup for codes below

import warnings
warnings.simplefilter('once', DeprecationWarning) 

import torch
from torch.nn import Parameter
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

model = [Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 0.1)
scheduler = StepLR(optimizer, 2)

for epoch in range(10):
    print(epoch, scheduler.get_lr()[0])
    optimizer.step()
    scheduler.step()

for epoch in range(10):
    print(epoch, scheduler.get_lr()[0])
    optimizer.step()
    scheduler.step()

for epoch in range(10):
    print(epoch, scheduler.get_last_lr()[0])
    optimizer.step()
    scheduler.step()

# Setup for codes below

import warnings
warnings.simplefilter('once', DeprecationWarning) 

import torch
from torch.nn import Parameter
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR

model = [Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 0.1)
scheduler = StepLR(optimizer, 2)

for epoch in range(10):
    optimizer.step()
    scheduler.step(epoch)
    print(epoch, optimizer.param_groups[0]['lr'])

for epoch in range(10):
    optimizer.step()
    scheduler.step(epoch)
    print(epoch, optimizer.param_groups[0]['lr'])

for epoch in range(10):
    print(epoch, optimizer.param_groups[0]['lr'])
    optimizer.step()
    scheduler.step()

for epoch in [0, 0, 1, 1, 2, 2, 3, 3]:
    optimizer.step()
    scheduler.step(epoch)
    print(epoch, optimizer.param_groups[0]['lr'])

for epoch in [0, 0, 1, 1, 2, 2, 3, 3]:
    optimizer.step()
    scheduler.step(epoch)
    print(epoch, optimizer.param_groups[0]['lr'])

for epoch in range(8):
    print(epoch, optimizer.param_groups[0]['lr'])

    optimizer.step()

    # Step at every other epoch
    if epoch % 2:
        scheduler.step()

optimizer = torch.optim.SGD(net.parameters(), 0.1)
lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)

import torch
from torch.nn import Parameter
from torch.optim import SGD

model = [Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 0.1)

# scheduler1 = ExponentialLR(optimizer, gamma=0.9)
scheduler1 = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=100)
scheduler2 = torch.optim.lr_scheduler.CyclicLR(optimizer, 0.1, 1, step_size_up=100)

values1, values2 = [], []
for epoch in range(1000):
    values1.append(scheduler2.get_last_lr()[0])
    optimizer.step()
    scheduler1.step(epoch)
    scheduler2.step()

import matplotlib.pyplot as plt
plt.plot(values1, label='values 1')
plt.legend()
plt.show()