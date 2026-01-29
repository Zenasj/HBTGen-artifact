# torch.rand(B, 40, 2048, 128, dtype=torch.bfloat16)  # Input shape for Llama-13B with sequence length 2k

import torch
import torch.nn as nn
import torch.nn.functional as F

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        
    def forward(self, x):
        with torch.backends.cuda.sdp_kernel(
            enable_flash=True,
            enable_math=False,
            enable_mem_efficient=False,
        ):
            return F.scaled_dot_product_attention(x, x, x, is_causal=True)

def my_model_function():
    # Return an instance of MyModel
    return MyModel()

def GetInput():
    # Return a random tensor input that matches the input expected by MyModel
    bsize = 1  # Example batch size, can be adjusted
    return torch.randn(bsize, 40, 2048, 128, dtype=torch.bfloat16).cuda()

