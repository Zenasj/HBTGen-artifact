import torch
from torch.nn import Parameter
from torch.optim import SGD
from torch.optim.lr_scheduler import WarmUpLR, ExponentialLR

model = [Parameter(torch.randn(2, 2, requires_grad=True))]
optimizer = SGD(model, 1.0)
scheduler1 = WarmUpLR(optimizer, warmup_factor=0.1, warmup_iters=5, warmup_method="constant")
scheduler2 = ExponentialLR(optimizer, gamma=0.9)

for epoch in range(10):
     print(epoch, scheduler2.get_last_lr()[0])
     optimizer.step()
     scheduler1.step()
     scheduler2.step()