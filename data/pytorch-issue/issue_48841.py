import torch
def sample_permutations(upper: int, indices: torch.Tensor, num_permutations: int):
    probas = torch.zeros((num_permutations, upper))
    probas[:, indices] = 1 / indices.shape[0]
    return torch.multinomial(probas, num_samples=indices.shape[0])

perturbable_inds = torch.tensor([ 1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29])
num_features = 41
num_samples = 1

for _ in range(10_000_000):
    sampled = sample_permutations(num_features, perturbable_inds, num_samples)
    if torch.any(torch.logical_or(sampled < 1, sampled > 29)):
        print(sampled)

import torch

print(torch.__version__)
probs = torch.zeros((10000,), device="cuda", dtype=torch.float16)
probs[0] = 1.0
for i in range(1000):
    value = torch.multinomial(probs, 1)
    if value != 0:
        print(i, value)
        break