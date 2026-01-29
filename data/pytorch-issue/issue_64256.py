import torch
import torch.nn as nn
import torch.nn.functional as F

# torch.rand(1, 2, dtype=torch.float64)  # Input shape (B=1, C=2)
class MyModel(nn.Module):
    def __init__(self, target, weight):
        super().__init__()
        self.register_buffer('target', target)
        self.register_buffer('weight', weight)

    def forward(self, input):
        # Compute cross_entropy loss (internally uses log_softmax + nll_loss)
        ce_loss = F.cross_entropy(input, self.target, weight=self.weight)
        # Directly compute nll_loss for explicit comparison
        log_probs = F.log_softmax(input, dim=1)
        nll_loss = F.nll_loss(log_probs, self.target, weight=self.weight)
        return ce_loss, nll_loss  # Return both losses for gradient comparison

def my_model_function():
    # Initialize with parameters that trigger the bug
    target = torch.randint(0, 2, (1,), dtype=torch.int64)
    weight = torch.tensor([1.0, -1.0], dtype=torch.float64)
    return MyModel(target=target, weight=weight)

def GetInput():
    # Generate input matching the shape required by MyModel
    return torch.randn((1, 2), dtype=torch.float64)

