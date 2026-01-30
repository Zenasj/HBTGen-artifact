import torch.nn as nn

import torch
probs = torch.tensor([0.1, 0.2, 0.3, 0.4])
N = 10_000
exp = torch.zeros(4)
for i in range(N):
    exp += torch.nn.functional.gumbel_softmax(probs.log(), tau=0.1, hard=True)
exp

tensor([ 961., 2046., 3051., 3942.])