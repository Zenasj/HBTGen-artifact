import torch
from torch.quasirandom import SobolEngine

seed = 123454321
x1 = SobolEngine(dimension=1, scramble=True, seed=seed).draw(3)
x2 = SobolEngine(dimension=1, scramble=True, seed=seed).draw(3)

assert torch.all(x1 == x2) ## succeeds

# seeded via torch.manual_seed
torch.manual_seed(seed)
x1 = SobolEngine(dimension=1, scramble=True).draw(3)
torch.manual_seed(seed)
x2 = SobolEngine(dimension=1, scramble=True).draw(3)

assert torch.all(x1 == x2) ## fails