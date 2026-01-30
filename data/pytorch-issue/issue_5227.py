import torch
import torch.distributions as dis
m = dis.Categorical(torch.Tensor([ 0.25, 0.25, 0.25, 0.25 ]))
m.sample()