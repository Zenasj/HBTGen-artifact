import torch.nn as nn

import torch
device = "cuda"
dtype = torch.float32
torch.manual_seed(42)

q = torch.normal(mean=0.0, std=((512 * 128) ** -0.5), size=[2, 4, 8, 128]).to(dtype=dtype, device=device)
k = torch.normal(mean=0.0, std=((512 * 128) ** -0.5), size=[2, 4, 16, 128]).to(dtype=dtype, device=device)
v = torch.normal(mean=0.0, std=((512 * 128) ** -0.5), size=[2, 4, 16, 128]).to(dtype=dtype, device=device)

enc_mask = torch.tensor([[1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0],
                              [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]], dtype=dtype, device=device)
dec_mask = torch.tensor([[1,1,1,1,1,1,1,1],
                              [1,1,1,1,1,1,0,0]], dtype=dtype, device=device)
cross_attention_mask = torch.einsum("be,bd->bed", dec_mask, enc_mask)
cross_attention_mask = cross_attention_mask[:, None, :, :]
cross_attention_mask = (1.0 - cross_attention_mask) * torch.finfo(dtype).min

with torch.backends.cuda.sdp_kernel(enable_mem_efficient=True):
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=cross_attention_mask)
    if torch.isnan(y).any() or torch.isinf(y).any():
        print("Invalid tensor for memory efficient!")
    else:
        print("Valid tensor for memory efficient!")

with torch.backends.cuda.sdp_kernel(enable_mem_efficient=False):
    y = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=cross_attention_mask)
    if torch.isnan(y).any() or torch.isinf(y).any():
        print("Invalid tensor for math!")
    else:
        print("Valid tensor for math!")