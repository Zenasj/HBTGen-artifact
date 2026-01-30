import torch
from torch.utils.data import WeightedRandomSampler

weights = torch.randn(10)
sampler = WeightedRandomSampler(weights=weights, num_samples=10)

self.weights = torch.tensor(weights, dtype=torch.double)