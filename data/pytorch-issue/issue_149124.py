# torch.rand(10, dtype=torch.float32)  # Example input shape (vocab_size,)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, num_neg_samples):
        super().__init__()
        self.num_neg_samples = num_neg_samples

    def forward(self, dist):
        # The problematic multinomial call that triggers CUDA assertion
        return torch.multinomial(dist, self.num_neg_samples, replacement=True)

def my_model_function():
    # Uses num_neg_samples=2 as in the original issue's example
    return MyModel(num_neg_samples=2)

def GetInput():
    # Generate a valid distribution tensor (sum to 1)
    vocab_size = 10  # Inferred from the example's word2idx size
    dist = torch.rand(vocab_size, dtype=torch.float32)
    dist /= dist.sum()  # Normalize to a probability distribution
    return dist

