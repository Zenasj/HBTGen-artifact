import torch

class MultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma, milestones, last_epoch = -1):
        self.init_lr = [group['lr'] for group in optimizer.param_groups]
        self.gamma = gamma
        self.milestones = milestones
        super().__init__(optimizer, last_epoch)

    def get_lr(self, *args):
        global_step = self.last_epoch #iteration number in pytorch
        gamma_power = ([0] + [i + 1 for i, m in enumerate(self.milestones) if global_step >= m])[-1]
        return [init_lr * (self.gamma ** gamma_power) for init_lr in self.init_lr]

optimizer = torch.optim.SGD([torch.rand(1)], lr = 1)
scheduler = MultiStepLR(optimizer, gamma = 1, milestones = [10, 20])

def my_scheduler(epoch):
  lr = 2**(-epoch)
  return lr

for i in range(len(optimizer.param_groups)):
  optimizer.param_groups[i]['lr'] = my_scheduler(epoch)

import torch

class MultiStepLR(torch.optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, gamma, milestones, last_epoch = -1):
        self.init_lr = [group['lr'] for group in optimizer.param_groups]
        self.gamma = gamma
        self.milestones = milestones
        super().__init__(optimizer, last_epoch)

    def get_lr(self):
        global_step = self.last_epoch #iteration number in pytorch
        gamma_power = ([0] + [i + 1 for i, m in enumerate(self.milestones) if global_step >= m])[-1]
        return [init_lr * (self.gamma ** gamma_power) for init_lr in self.init_lr]

optimizer = torch.optim.SGD([torch.rand(1)], lr = 1)
scheduler = MultiStepLR(optimizer, gamma = 1, milestones = [10, 20])

import torch                                                                                                                                                             
from torch.nn import Parameter 
from torch.optim import SGD 
from torch.optim.lr_scheduler import LambdaLR, MultiplicativeLR

model = [Parameter(torch.randn(2, 2, requires_grad=True))] 
optimizer = SGD(model, lr=1.)                                                                                                                                            

func = lambda epoch: 2.                                                                                                                 
scheduler = LambdaLR(optimizer, func)

for epoch in range(10): 
  print(epoch, scheduler.get_last_lr()[0]) 
  optimizer.step() 
  scheduler.step()

model = [Parameter(torch.randn(2, 2, requires_grad=True))] 
optimizer = SGD(model, lr=1.)                                                                                                                                            

func = lambda epoch: 2.                                                                                                                 
scheduler = MultiplicativeLR(optimizer, func)

for epoch in range(10): 
  print(epoch, scheduler.get_last_lr()[0]) 
  optimizer.step() 
  scheduler.step()