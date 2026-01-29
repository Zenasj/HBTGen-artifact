# torch.rand(B, 1, H, W, device='cuda')  # B=1, H=8, W=8
import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.register_buffer('q', torch.rand(1, 1, 8, 8, device='cuda'))
        self.register_buffer('k', torch.rand(1, 1, 8, 8, device='cuda'))
        self.register_buffer('v', torch.rand(1, 1, 8, 8, device='cuda'))
    
    def forward(self, mask):
        float_mask = mask.float()
        bool_mask = mask.bool()
        out_float = F.scaled_dot_product_attention(self.q, self.k, self.v, float_mask)
        out_bool = F.scaled_dot_product_attention(self.q, self.k, self.v, bool_mask)
        # Check for differences including NaN propagation
        diff = torch.any(
            (out_float != out_bool) | (out_float.isnan() ^ out_bool.isnan())
        )
        return diff  # Returns True if outputs differ (including NaNs)

def my_model_function():
    return MyModel()

def GetInput():
    B, H, W = 1, 8, 8
    mask = torch.rand(B, 1, H, W, device='cuda')
    mask[..., 4:, :] = 0
    mask[..., :, 4:] = 0
    mask[..., 4:, 4:] = 0
    return mask

