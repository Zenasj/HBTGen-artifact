py
import torch
dist = torch.distributions.Normal(0,1)
dist.enumerate_support()