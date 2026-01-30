import torch
import numpy as np

input = torch.tensor([0.5250, 0.5250, 0.5250, 0.5250, 0.5250, 0.5250])
print(torch.std(input)) # this gives 6.5294e-08 and not ZERO!!! on the cpu

input = torch.tensor([0.5250, 0.5250, 0.5250, 0.5250, 0.5250, 0.5250]).cuda()
print(torch.std(input)) # this gives zero! on the gpu

# even more surprisingly
input = torch.tensor([0.5251, 0.5251, 0.5251, 0.5251, 0.5251, 0.5251]) # off by one
print(torch.std(input)) # this gives zero now!

input = torch.tensor([0.5249, 0.5249, 0.5249, 0.5249, 0.5249, 0.5249]) # off by one
print(torch.std(input)) # this gives zero now!

tensor(6.5294e-08)
tensor(0., device='cuda:0')
tensor(0.)
tensor(0.)