import torch.nn as nn

import torch
from torch import nn

import gc

def main():
    param = nn.Parameter(torch.randn(10))

    optim = torch.optim.Adam([param])
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, lambda epoch: 1.0)
    del scheduler

    print(gc.get_referrers(optim))
    
    gc.collect()
    del optim
    print(gc.collect())

if __name__ == '__main__':
    main()

def _update_last_epoch(epoch=None):
      if epoch is None:
          epoch = self.last_epoch + 1
      self.last_epoch = epoch

for param_group, lr in zip(self.optimizer.param_groups, self.get_lr()):
      param_group['lr'] = lr