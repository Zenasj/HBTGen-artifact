# torch.nested.nested_tensor([torch.rand(seq_len, 1024) for seq_len in [random lengths]], layout=torch.jagged, device='cuda') ← Input shape: (batch_size, j1, 1024)
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

class MyModel(nn.Module):
    def __init__(self, embed_dim=1024, num_heads=8):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.proj_q = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_k = nn.Linear(embed_dim, embed_dim, bias=False)
        self.proj_v = nn.Linear(embed_dim, embed_dim, bias=False)
        
        self.lnorm_q = nn.LayerNorm(embed_dim, elementwise_affine=False)
        self.lnorm_k = nn.LayerNorm(embed_dim, elementwise_affine=False)

    def forward(self, x):
        q = self.proj_q(x)
        k = self.proj_k(x)
        v = self.proj_v(x)
        
        q = self.lnorm_q(q)
        k = self.lnorm_k(k)
        
        # Split into heads
        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        k = k.unflatten(-1, (self.num_heads, self.head_dim))
        v = v.unflatten(-1, (self.num_heads, self.head_dim))
        
        # Transpose to (batch, heads, seq, head_dim)
        q = q.transpose(-2, -3)
        k = k.transpose(-2, -3)
        v = v.transpose(-2, -3)
        
        # Apply scaled dot product attention with flash
        with torch.backends.cuda.sdp_kernel(enable_flash=True):
            attn_output = F.scaled_dot_product_attention(
                q, k, v,
                dropout_p=0.0,
                is_causal=False
            )
        # Reshape back
        attn_output = attn_output.transpose(-2, -3)
        attn_output = attn_output.flatten(start_dim=-2)
        
        return attn_output

def my_model_function():
    # Return an instance of MyModel on CUDA
    return MyModel().to('cuda')

def GetInput():
    # Return a random nested tensor input (CUDA)
    batch_size = 10
    embed_dim = 1024
    seq_lens = [random.randint(10, 20) for _ in range(batch_size)]
    tensors = [torch.randn(seq_len, embed_dim, device='cuda') for seq_len in seq_lens]
    return torch.nested.nested_tensor(tensors, layout=torch.jagged, device='cuda')

