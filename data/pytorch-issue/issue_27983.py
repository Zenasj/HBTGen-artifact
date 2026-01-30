import torch
from torch.distributions.categorical import Categorical

dev = torch.device('cuda')
logits = torch.randn(2, 0, 4, 5, device=dev)
dist = Categorical(logits=logits)
sample = dist.sample()
print(sample.shape)