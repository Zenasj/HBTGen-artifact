# torch.rand(2, 7, dtype=torch.bool) ← inferred input shape (batch_size=2, n=7)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.x = nn.Parameter(torch.rand(3, 7))  # b=3, n=7

    def forward(self, masks):
        # masks: (batch_size, 7) boolean tensor
        # Compute x[:, mask].sum(-1) for each mask in batch
        batch_size = masks.size(0)
        x_expanded = self.x.unsqueeze(0).expand(batch_size, -1, -1)  # (batch_size, 3, 7)
        mask_float = masks.to(dtype=x_expanded.dtype).unsqueeze(1)  # (batch_size, 1, 7)
        selected = x_expanded * mask_float
        summed = selected.sum(dim=-1)  # (batch_size, 3)
        return summed

def my_model_function():
    return MyModel()

def GetInput():
    # Returns a (2,7) boolean tensor (transposed partition)
    partition = torch.rand(7, 2) > 0.5
    return partition.T  # (2,7)

