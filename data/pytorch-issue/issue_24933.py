import numpy as np

class BCELoss(nn.Module, _Logger):
    __name__ = 'bce_loss'

    def __init__(self, with_logits=True, **kwargs):
        super().__init__()
        self.with_logits = with_logits
        if self.with_logits:
            self.bce_loss = torch.nn.BCEWithLogitsLoss(**kwargs)
        else:
            self.bce_loss = torch.nn.BCELoss(**kwargs)

    def forward(self, inputs, targets, logs=None):
        loss = self.bce_loss(inputs, targets)
        self.update_logs(logs, loss)
        return loss


class BCEAndDiceLoss(nn.Module):
    __name__ = 'bce_and_dice_loss'

    def __init__(self, with_logits=True, dice_loss_kwargs={}, bce_kwargs={}, _dice_loss_class=None):
        super().__init__()
        self.with_logits = with_logits
        if _dice_loss_class is None:
            _dice_loss_class = DiceLoss
        self.dice_loss = _dice_loss_class(
            with_logits=self.with_logits, **dice_loss_kwargs)
        self.bce_loss = BCELoss(with_logits=self.with_logits, **bce_kwargs)

    def forward(self, inputs, targets, logs=None, dice_loss_kwargs={}, bce_kwargs={}):
        d_loss = self.dice_loss(inputs, targets, logs=logs, **dice_loss_kwargs)
        b_loss = self.bce_loss(inputs, targets, logs=logs, **bce_kwargs)
        loss = b_loss + d_loss
        if not np.all(loss.cpu().data.numpy() < 1e4):
            import pickle
            with open('error_tensors.pkl', 'wb') as out:
                pickle.dump((inputs.cpu().detach(),
                             targets.cpu().detach(),
                             (b_loss.cpu().detach(),
                              BCELoss(with_logits=False,
                                      **bce_kwargs)(nn.Sigmoid()(inputs), targets).cpu().detach()
                                )), out)
        assert np.all(loss.cpu().data.numpy() < 1e4), (d_loss, BCELoss(
                                                           with_logits=True, **bce_kwargs)(
                                                               inputs, targets),

                                                       BCELoss(
                                                           with_logits=False, **bce_kwargs)(inputs, targets),
                                                       inputs.max(), inputs.min(), inputs.mean(),
                                                       targets.max(), targets.min(), targets.mean())
        return loss

(tensor(0.9884, device='cuda:0'), tensor(105739.7031, device='cuda:0'), tensor(1.8102, device='cuda:0'), tensor(391591.2188, device='cuda:0'), tensor(-2.1709e+08, device='cuda:0'), tensor(-6036326., device='cuda:0'), tensor(1., device='cuda:0'), tensor(0., device='cuda:0'), tensor(0.0093, device='cuda:0'))

import torch
import torch.nn as nn

def compare_bce_sigmoid_vs_logits(device):
  use_grad = False
  sizes = [2**exponent for exponent in range(10, 15)]

  print("tensor_size (x,y): mean_logits_loss, mean_sigmoid_loss, mean_loss_diff")
  print("---------")
  for y in sizes:
    for x in sizes:
      bce_logits_losses = []
      bce_sigmoid_losses = []
      for _ in range(5):
        target = torch.rand(x, y, requires_grad=use_grad, dtype=torch.float32, device=device)
        output = target - 0.5

        bce_logits_losses.append(nn.BCEWithLogitsLoss()(output, target))
        bce_sigmoid_losses.append(nn.BCELoss()(torch.sigmoid(output), target))

      bce_logits_losses = torch.tensor(bce_logits_losses, device=device)
      bce_sigmoid_losses = torch.tensor(bce_sigmoid_losses, device=device)
      loss_diffs = (bce_logits_losses - bce_sigmoid_losses).abs()

      logits_std, logits_mean = torch.std_mean(bce_logits_losses)
      sigmoid_std, sigmoid_mean = torch.std_mean(bce_sigmoid_losses)
      diff_std, diff_mean = torch.std_mean(loss_diffs)

      print('(%6d, %6d): %f +/-%.2e, %f +/-%.2e, %f +/-%.2e' % (
        x, y,
        logits_mean, 2*logits_std,
        sigmoid_mean, 2*sigmoid_std,
        diff_mean, 2*diff_std
      ))

    print()

print('=========')
print('CUDA:')
print('=========')
compare_bce_sigmoid_vs_logits(torch.device('cuda'))
print()
print()
print('=========')
print('CPU:')
print('=========')
compare_bce_sigmoid_vs_logits(torch.device('cpu'))

def compare_sigmoid_cuda_vs_cpu():
  device_cpu = torch.device('cpu')
  device_cuda = torch.device('cuda')

  sizes = [2**exponent for exponent in range(10, 15)]
  for y in sizes:
    for x in sizes:
      t_cpu = torch.rand(x, y, requires_grad=False, dtype=torch.float32, device=device_cpu)
      t_cuda = t_cpu.clone().detach().to(device=device_cuda)

      sig_cpu = torch.sigmoid(t_cpu)
      sig_cuda = torch.sigmoid(t_cuda)

      mean_diff = (sig_cuda - sig_cpu.to(device=device_cuda)).abs().max()

      print('(%6d, %6d): %f' % (x, y, mean_diff))
    print()

nn.BCELoss(reduction='none')(torch.sigmoid(output), target).mean()

import torch
import torch.nn as nn
target = torch.tensor([0., 1., 1., 0., 0.5, 0.5], dtype=torch.float)
output = torch.tensor([-100., 100., -100., 100., -100., 100.], dtype=torch.float)
print('sigmoid(output): %s' % str(torch.sigmoid(output)))
print('target: %s' % str(target))
print('BCELoss: %s' % str(
    nn.BCELoss(reduction='none')(torch.sigmoid(output), target)
))
print('BCEWithLogitsLoss: %s' % str(
    nn.BCEWithLogitsLoss(reduction='none')(output, target)
))