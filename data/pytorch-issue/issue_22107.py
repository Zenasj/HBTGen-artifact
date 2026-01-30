import torchvision

python
import torch
from torchvision.models import resnet18
net = resnet18()
optimizer = torch.optim.SGD(net.parameters(), 0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
for i in range(10):
    print(i, scheduler.get_lr())
    scheduler.step()

0 [0.1]
1 [0.1]
2 [0.1]
3 [0.0010000000000000002]
4 [0.010000000000000002]
5 [0.010000000000000002]
6 [0.00010000000000000003]
7 [0.0010000000000000002]
8 [0.0010000000000000002]
9 [1.0000000000000004e-05]

python
def step(self, epoch=None):
    if epoch is None:
        epoch = self.last_epoch + 1
    self.last_epoch = epoch
    for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
        param_group['lr'] = lr

python
def get_lr(self):
    if self.last_epoch not in self.milestones:
        return [group['lr'] for group in self.optimizer.param_groups]
    return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
            for group in self.optimizer.param_groups]

import torch
from torch.nn import Parameter
optimizer = torch.optim.SGD([Parameter(torch.randn(2, 2, requires_grad=True))], 0.1)
scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[3, 6, 9], gamma=0.1)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 3, gamma=0.1)
for i in range(10):
    print(i, scheduler.get_lr())
    scheduler.step()

python
class MultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma=0.1, last_epoch=-1):
        self.milestones = Counter(milestones)
        self.gamma = gamma
        super(MultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch not in self.milestones:
            return [group['lr'] for group in self.optimizer.param_groups]
        return [group['lr'] * self.gamma ** self.milestones[self.last_epoch]
                for group in self.optimizer.param_groups]