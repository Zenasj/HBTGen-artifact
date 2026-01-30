import torch
from torch import optim
from torch.utils import data

X = torch.arange(16) 
dataset = data.TensorDataset(X) 

optimizer = optim.Adam([torch.rand(10, 20),], lr=1e-4) 

warmup_factor = 2 
decay_factor = 0.5 

warmup_schedular = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : warmup_factor ** epoch)  

batchSize = 16
dataloader = data.DataLoader(dataset, batch_size=batchSize) 

schedular = warmup_schedular 

for epoch in range(4): 
  if epoch == 2: 
    schedular = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : decay_factor ** epoch) 
  for x in dataloader: 
      optimizer.step() 
  schedular.step() 
  print(optimizer.param_groups[0]['lr'])

import torch
from torch import optim
from torch.utils import data

X = torch.arange(16) 
dataset = data.TensorDataset(X) 

optimizer = optim.Adam([torch.rand(10, 20),], lr=2e-4) 

warmup_factor = 2 
decay_factor = 0.5 

scheduler1 = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch : warmup_factor)  
scheduler2 = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch : decay_factor)
batchSize = 16
dataloader = data.DataLoader(dataset, batch_size=batchSize) 

for epoch in range(4): 
  print(optimizer.param_groups[0]['lr']) 
  for x in dataloader: 
      optimizer.step() 
  if epoch <= 0:
    scheduler1.step()
  else:
    scheduler2.step()

import torch
from torch import optim
from torch.utils import data

X = torch.arange(16) 
dataset = data.TensorDataset(X) 

optimizer = optim.Adam([torch.rand(10, 20),], lr=2e-4) 

warmup_factor = 2 
decay_factor = 0.5 

scheduler1 = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : warmup_factor**epoch)
scheduler2 = optim.lr_scheduler.LambdaLR(optimizer, lambda epoch : 1. if epoch <=0 else decay_factor ** (epoch-1))

batchSize = 16
dataloader = data.DataLoader(dataset, batch_size=batchSize) 

for epoch in range(4): 
  print(optimizer.param_groups[0]['lr']) 
  for x in dataloader: 
      optimizer.step() 
  if epoch <= 0:
    scheduler1.step()
  else:
    scheduler2.step()

import torch
from torch import optim
from torch.utils import data

X = torch.arange(16) 
dataset = data.TensorDataset(X) 

optimizer = optim.Adam([torch.rand(10, 20),], lr=2e-4) 

warmup_factor = 2 
decay_factor = 0.5 

scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lambda epoch : warmup_factor if epoch <= 1 else decay_factor)

batchSize = 16
dataloader = data.DataLoader(dataset, batch_size=batchSize) 

for epoch in range(4): 
  print(optimizer.param_groups[0]['lr']) 
  for x in dataloader: 
      optimizer.step() 
  scheduler.step()