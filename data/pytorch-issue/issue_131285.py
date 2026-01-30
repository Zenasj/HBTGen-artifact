attn_weights_l.masked_fill_(attention_mask_v, float("-inf"))
# to 
attn_weights_l.masked_fill_(attention_mask_v.contiguous(), float("-inf"))

import torch

device = "mps"
attn_weights = torch.load("attn_weights.pt").to(device)
attn_mask = torch.load("attn_mask.pt").to(device)

print(attn_weights.shape)
print(attn_mask.shape)
print(attn_mask.sum())

attn_weights.masked_fill_(attn_mask, float("-inf"))

print((attn_weights == float("-inf")).sum())

attn_weights[0][0][0] = float("-inf")

print((attn_weights == float("-inf")).sum())