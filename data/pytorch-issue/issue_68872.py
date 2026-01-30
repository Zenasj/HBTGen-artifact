import torch.nn as nn

import torch
loss_fn = torch.nn.NLLLoss()
inputs, targets = next(iter(dataloader))
device = 'cuda' if torch.cuda.is_available() else 'cpu'
inputs, targets = inputs.to(device), targets.to(device)
model = model.to(device)
logits = model(inputs)
loss = loss_fn(logits, targets, reduction='sum').item()

py
loss_fn = torch.nn.NLLLoss(reduction='sum')
...