import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import (
    create_block_mask,
    flex_attention,
)

torch.set_default_device("cuda")
torch.manual_seed(0)

torch._dynamo.config.cache_size_limit = 1000

# Compile the flex_attention function
flex_attention = torch.compile(flex_attention, dynamic=False)


data_type = torch.bfloat16

def get_x_y_z(idx):
    z = idx // (spatial_size * spatial_size)
    remaining = idx % (spatial_size * spatial_size)
    y = remaining // spatial_size
    x = remaining % spatial_size
    return x, y, z

def natten_mask_3d(b, h, q_idx, kv_idx):
    q_x, q_y, q_z = get_x_y_z(q_idx)
    kv_x, kv_y, kv_z = get_x_y_z(kv_idx)
    
    # Kernel centers, clamped to fixed distances from volume edges
    kernel_x = q_x.clamp(kernel_size // 2, (spatial_size - 1) - kernel_size // 2)
    kernel_y = q_y.clamp(kernel_size // 2, (spatial_size - 1) - kernel_size // 2)
    kernel_z = q_z.clamp(kernel_size // 2, (spatial_size - 1) - kernel_size // 2)
    
    # Check if key/value is within the kernel in all three dimensions
    hori_mask = (kernel_x - kv_x).abs() <= kernel_size // 2
    vert_mask = (kernel_y - kv_y).abs() <= kernel_size // 2
    depth_mask = (kernel_z - kv_z).abs() <= kernel_size // 2
    
    return hori_mask & vert_mask & depth_mask

spatial_size = 48
kernel_size = 7
S = spatial_size ** 3
B = 1
H = 16
D = 64
query = torch.randn(
    B, H, S, D, device="cuda", dtype=data_type, requires_grad=True
)
key = torch.randn(
    B, H, S, D, device="cuda", dtype=data_type, requires_grad=True
)
value = torch.randn(
    B, H, S, D, device="cuda", dtype=data_type, requires_grad=True
)
block_mask = create_block_mask(natten_mask_3d, B=1, H=1, Q_LEN=S, KV_LEN=S, device="cuda", _compile=True, BLOCK_SIZE=4096)
x = flex_attention(query, key, value, block_mask=block_mask)