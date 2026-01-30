import torch
import torch.nn as nn

my_attn = nn.MultiheadAttention(
    embed_dim=256,
    batch_first=True,
    num_heads=8
)
my_attn = my_attn.to("cuda")

x = torch.randn((1, 10, 256), dtype=torch.float, device="cuda")
bool_mask = torch.ones((10, 10), dtype=torch.bool, device="cuda")
bool_mask[:2, :2] = False
bool_mask[2:, 2:] = False

float_mask = torch.randn((10, 10), dtype=torch.float, device="cuda")

my_attn.eval()
with torch.no_grad():
    bool_mask_res, bool_mask_scores = my_attn(
        query=x, key=x, value=x, attn_mask=bool_mask
    )
    print(bool_mask_scores)
    
    float_mask_res, float_mask_scores = my_attn(
        query=x, key=x, value=x, attn_mask=float_mask
    )
    print(float_mask_scores)       # NOTE: the bug is here, the scores will be all 'nan'.

import torch
import torch.nn as nn

my_attn = nn.MultiheadAttention(
    embed_dim=256,
    batch_first=True,
    num_heads=8
)
my_attn = my_attn.to("cuda")

x = torch.randn((1, 10, 256), dtype=torch.float, device="cuda")
bool_mask = torch.ones((10, 10), dtype=torch.bool, device="cuda")
bool_mask[:2, :2] = False
bool_mask[2:, 2:] = False

float_mask = torch.randn((10, 10), dtype=torch.float, device="cuda")

my_attn.eval()

with torch.no_grad():
    bool_mask_res, bool_mask_scores = my_attn(
        query=x.clone(), key=x, value=x, attn_mask=bool_mask    # Just use .clone()
    )
    print(bool_mask_scores)
    
    float_mask_res, float_mask_scores = my_attn(
        query=x.clone(), key=x, value=x, attn_mask=float_mask    # Just use .clone()
        #
    )
    print(float_mask_scores)