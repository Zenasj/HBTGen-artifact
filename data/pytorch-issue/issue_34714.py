# torch.rand(B, C, dtype=torch.float32)  # e.g., B=2, C=2**16
import torch
from torch import nn

class FixedCategorical(torch.distributions.Categorical):
    def sample(self, sample_shape=torch.Size()):
        if not isinstance(sample_shape, torch.Size):
            sample_shape = torch.Size(sample_shape)
        probs_2d = self.probs.reshape(-1, self._num_events)
        samples_2d = torch.multinomial(probs_2d, sample_shape.numel(), True).T
        return samples_2d.reshape(self._extended_shape(sample_shape))

class MyModel(nn.Module):
    def __init__(self, sample_shape=torch.Size([2**16])):
        super().__init__()
        self.sample_shape = sample_shape

    def forward(self, probs):
        fixed_dist = FixedCategorical(probs)
        return fixed_dist.sample(self.sample_shape)

def my_model_function():
    return MyModel()

def GetInput():
    # Example input from the issue: shape (2, 2**16)
    B, C = 2, 2**16
    return torch.rand(B, C, dtype=torch.float32)

