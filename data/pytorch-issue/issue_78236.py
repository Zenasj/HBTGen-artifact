from torch.utils.data import WeightedRandomSampler
import torch

weights = torch.rand(800,1)+1e-3
sampler =  WeightedRandomSampler(weights, num_samples=32)
results = list(sampler)
for samples in results:
    assert all([sample == 0 for sample in samples])
# reaches this location without raising assertions