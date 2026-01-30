import torch

from torch.optim.lr_scheduler import _LRScheduler
class myMultiStepLR(_LRScheduler):
    def __init__(self, optimizer, milestones, gamma, last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError('Milestones should be a list of'
                             ' increasing integers. Got {}', milestones)
        self.milestones = milestones
        self.gamma = gamma
        super(myMultiStepLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        if self.last_epoch in self.milestones:
            index = self.milestones.index(self.last_epoch)
            _gamma = self.gamma[index]
        else:
            _gamma = 1.0
        return [base_lr * _gamma for base_lr in self.base_lrs]