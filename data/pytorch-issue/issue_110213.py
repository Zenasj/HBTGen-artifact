# torch.rand(1, 1, 3, 4, dtype=torch.float32)  # Example input shape for q (similar for k and v)
import torch
from torch import nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def forward(self, inputs):
        q, k, v, mask = inputs
        with torch.backends.cuda.sdp_kernel(enable_math=True, enable_flash=False, enable_mem_efficient=False):
            out_math = F.scaled_dot_product_attention(q, k, v, mask, is_causal=False)
        
        with torch.backends.cuda.sdp_kernel(enable_math=False, enable_flash=True, enable_mem_efficient=True):
            out_fused = F.scaled_dot_product_attention(q, k, v, mask, is_causal=False)
        
        # Return True if fused output has NaN and math does not
        fused_has_nan = torch.isnan(out_fused).any()
        math_has_nan = torch.isnan(out_math).any()
        return fused_has_nan and not math_has_nan

def my_model_function():
    return MyModel()

def GetInput():
    B, H, seq_len, head_dim = 1, 1, 3, 4
    q = torch.randn(B, H, seq_len, head_dim, dtype=torch.float32)
    k = torch.randn(B, H, seq_len, head_dim, dtype=torch.float32)
    v = torch.randn(B, H, seq_len, head_dim, dtype=torch.float32)
    neg_value = torch.finfo(q.dtype).min
    mask = torch.tensor([[[[neg_value, neg_value, neg_value],
                          [1.0, 1.0, 1.0],
                          [1.0, 1.0, 1.0]]]], dtype=q.dtype)
    return (q, k, v, mask)

