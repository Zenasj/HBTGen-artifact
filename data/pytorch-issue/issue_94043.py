# torch.rand(2, 10, dtype=torch.float32)
import torch
from torch import nn

class MyModel(nn.Module):
    def forward(self, x):
        # Compute original MPS log_softmax (problematic implementation)
        original = x.log_softmax(dim=-1)
        # Compute custom numerically stable log_softmax
        c = x.max(dim=-1, keepdim=True).values
        logsumexp = torch.log(torch.exp(x - c).sum(dim=-1, keepdim=True))
        custom = x - c - logsumexp
        # Compare outputs using 5e-5 threshold from issue's validation
        diff = torch.abs(original - custom)
        return torch.all(diff < 5e-5)  # Returns boolean tensor indicating if all elements are within tolerance

def my_model_function():
    return MyModel()

def GetInput():
    return torch.rand(2, 10, dtype=torch.float32)

