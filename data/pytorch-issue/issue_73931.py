py
import torch
dist = torch.distributions
dist.Normal(0.5, 1).sample(100)