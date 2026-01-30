import torch
import torch.nn as nn

linear1 = nn.Linear(1,1)
seq = nn.Sequential(linear1)
seq.linear2 = nn.Linear(2,2)
seq(torch.tensor([[0.5]]))  # raises error because it tries to forward seq.linear2 after linear1