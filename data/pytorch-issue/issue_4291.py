import torch.nn.functional as F

def crit_fn(input, target):
  x = F.mse_loss(input, target, reduction='none')
  x = x.mean()
  return x

learn.crit = crit_fn