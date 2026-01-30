import torch
import torch.nn.functional as F

from torch.nn import functional as F

float_min = torch.finfo(torch.float16).min

attention_mask_fp16 = (attention_mask * 1.0).masked_fill(attention_mask, float_min).to(torch.float16)

attn_output = F.scaled_dot_product_attention(
                 query_layer_, key_layer_, value_layer_, attention_mask_fp16, 0.0, is_causal=False
 )