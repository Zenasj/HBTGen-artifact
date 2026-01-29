# torch.randint(0, 27, (B, 3), dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(2147483647)
        self.C = nn.Parameter(torch.randn((27, 10), generator=g))
        self.W1 = nn.Parameter(torch.randn((10 * 3, 100), generator=g))
        self.b1 = nn.Parameter(torch.randn(100, generator=g) * 0.1)

    def forward(self, X):
        emb = self.C[X].flatten(1)
        logits = emb @ self.W1 + self.b1
        logit_maxes = logits.max(dim=1, keepdim=True).values
        norm_logits = logits - logit_maxes
        counts = norm_logits.exp()
        counts_sum = counts.sum(1, keepdim=True)
        counts_sum_inv = counts_sum ** -1
        probs = counts * counts_sum_inv
        logprobs = probs.log()
        return logprobs

def my_model_function():
    return MyModel()

def GetInput():
    return torch.randint(0, 27, (32, 3), dtype=torch.long)

