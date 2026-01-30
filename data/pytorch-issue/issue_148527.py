import torch.nn as nn

from torch.nn.attention.flex_attention import create_block_mask, flex_attention
import torch

flex_attention = torch.compile(flex_attention, dynamic=False)

q = torch.randn(1, 32, 8, 128, dtype=torch.bfloat16, device="cuda:0")
k = torch.randn(1, 32, 8, 128, dtype=torch.bfloat16, device="cuda:0")
v = torch.randn(1, 32, 8, 128, dtype=torch.bfloat16, device="cuda:0")

def easy_head_attention_mod(head_num):
    head_type = torch.tensor([False if i % head_num == 0 else True for i in range(head_num)], dtype=torch.bool, device=q.device)

    def mask_mod(b, h, q_idx, kv_idx):
        bi_mask = True & head_type[h]
        causal_mask = q_idx >= kv_idx

        return bi_mask & causal_mask

    return mask_mod


mask_mod = easy_head_attention_mod(32) # Error occurs when head_num is greater than 1, e.g., 32
# If head_num is set to 1 (e.g., mask_mod = easy_head_attention_mod(1)), the code runs without error

mask = create_block_mask(mask_mod, 1, 32, 8, 8, device=q.device, _compile=True)

# Use `enable_gqa=True` with corresponding inputs here would bring more bugs
attn_output = flex_attention(q, k, v, block_mask=mask)

print(attn_output.shape)