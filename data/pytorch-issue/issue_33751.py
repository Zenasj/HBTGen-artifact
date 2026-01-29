# torch.rand(3, 3, 4, dtype=torch.long)
import torch
from torch import nn

class MyModel(nn.Module):
    def __init__(self, B=3, H=3, W=4):
        super().__init__()
        self.B = B
        self.H = H
        self.W = W
        # Precompute indices and values as buffers for device handling
        batch_idx = torch.arange(B).repeat_interleave(H * W).long()
        hw_idx = torch.ones_like(batch_idx)
        self.register_buffer('batch_idx', batch_idx)
        self.register_buffer('hw_idx_h', hw_idx)  # H dimension indices
        self.register_buffer('hw_idx_w', hw_idx)  # W dimension indices
        # Values must match indices' numel (B*H*W) for GPU compatibility
        self.register_buffer('values', torch.ones(B * H * W, dtype=torch.long))

    def forward(self, grid):
        indices = (self.batch_idx, self.hw_idx_h, self.hw_idx_w)
        return grid.index_put_(indices, self.values, accumulate=True)

def my_model_function():
    return MyModel()

def GetInput():
    return torch.zeros(3, 3, 4, dtype=torch.long)

