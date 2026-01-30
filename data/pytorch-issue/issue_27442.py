import torch.nn as nn

from torch import nn
import torch
import math
import time

time_step  = 4000  # Input sequence length
vocab_size = 3  # Number of classes
batch_size = 4  # Batch size
target_sql = 700  # Target sequence length

ctc_loss  = nn.CTCLoss(reduction='sum')

for j in range(10):

  print('\nLogits length : ',time_step,'Label length : ',target_sql)
  print('-------------------')
  
  x = torch.randn(time_step, batch_size, vocab_size).requires_grad_().cuda()#.double() #uncomment to use double
  x = nn.Parameter(x)
  y = torch.randint(low=1, high=vocab_size-1, size=(batch_size, target_sql),dtype=torch.long).cuda()

  x_lengths = torch.full(size=(batch_size,), fill_value=time_step, dtype=torch.long).cuda()
  y_lengths = torch.full(size=(batch_size,), fill_value=target_sql, dtype=torch.long).cuda()

  loss1 = ctc_loss(x, y, x_lengths, y_lengths)
  loss2 = ctc_loss(x.cpu(), y.cpu(), x_lengths.cpu(), y_lengths.cpu()) 
  
  loss1.backward()
  tg1 = x.grad.clone().detach()
  x.grad.zero_()

  loss2.backward()
  tg2 = x.grad.clone().detach()
  x.grad.zero_()

  print('Grads  Diff : ',torch.norm(tg1-tg2).item())
  print('Losses Diff : ',torch.norm(loss1.cpu()-loss2).item())

  target_sql += 100