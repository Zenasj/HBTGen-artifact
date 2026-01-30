import torch

from torch import Tensor, distributions as tdist

def draw_three_samples(d: tdist.Distribution) -> Tensor:
    d.sample((3,))
    return d.rsample((3,))

def sample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
    ...

def rsample(self, sample_shape: torch.Size = torch.Size()) -> torch.Tensor:
    ...