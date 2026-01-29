# torch.rand(B, 256, 96, dtype=torch.float32)
import torch
import numpy as np
from torch import nn

class MyModel(nn.Module):
    def __init__(self, H=16, W=16, window_size=7, shift_size=3, C=96):
        super().__init__()
        self.H = H
        self.W = W
        self.window_size = window_size
        self.shift_size = shift_size
        self.C = C  # Channel dimension, inferred from input

        # Precompute static Hp and Wp to avoid dynamic shape issues
        self.Hp = int(np.ceil(H / window_size)) * window_size
        self.Wp = int(np.ceil(W / window_size)) * window_size

    def forward(self, x):
        # x: (B, H*W, C)
        device = x.device

        # Create mask with precomputed static Hp/Wp
        img_mask = torch.zeros((1, self.Hp, self.Wp, 1), device=device)
        h_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        w_slices = (
            slice(0, -self.window_size),
            slice(-self.window_size, -self.shift_size),
            slice(-self.shift_size, None)
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1
        return img_mask  # Example output, replace with actual model logic

def my_model_function():
    # Uses default H=16, W=16, window_size=7 (common Swin parameters)
    return MyModel()

def GetInput():
    # Matches the input shape (B, H*W, C) with H=16, W=16, C=96
    B = 2
    return torch.rand(B, 16*16, 96, dtype=torch.float32)

