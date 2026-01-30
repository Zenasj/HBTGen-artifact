import torch.nn as nn

import torch
from torch import nn
import torch.nn.functional as F

class M(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.all_head_size = 12 * 64
        self.dense = nn.Linear(self.all_head_size, self.all_head_size)

    def forward(self, q, k, v):
        context_layer = F.scaled_dot_product_attention(
            q, k, v, attn_mask=None, dropout_p=0.2
        )
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(new_context_layer_shape)
        return self.dense(context_layer)
 
 
if __name__ == "__main__":
    mod = M().to(torch.bfloat16).eval()
 
    q = torch.randn((28, 12, 512, 64), dtype=torch.bfloat16)
    k = torch.randn((28, 12, 512, 64), dtype=torch.bfloat16)
    v = torch.randn((28, 12, 512, 64), dtype=torch.bfloat16)
    inputs = (q, k, v,)

    is_inference = 0

    if is_inference:
        # 1. INFERENCE: run successfully
        with torch.no_grad(), torch.cpu.amp.autocast():
            compiler_mode = torch.compile(mod)
            _ = compiler_mode(*inputs)
            output = compiler_mode(*inputs)
    else:
        # 2. TRAINING: runtime error: mat1 and mat2 must have the same dtype, but got Float and BFloat16
        with torch.cpu.amp.autocast():
            compiler_mode = torch.compile(mod)
            _ = compiler_mode(*inputs)
            output = compiler_mode(*inputs)