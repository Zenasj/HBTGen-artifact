import torch

labels = torch.load('labels.pt').to('cuda:0') # labels.pt is attached
logits = torch.ones(190, 250002, 50).float().to('cuda:0')
loss = torch.nn.functional.cross_entropy(logits, labels)

import torch
import torch.nn as nn

labels = torch.zeros(190, 50, dtype=torch.long, device="cuda")

logits = torch.ones(190, 229000, 50).float().to('cuda:0')
loss = torch.nn.functional.cross_entropy(logits, labels)
print(loss)