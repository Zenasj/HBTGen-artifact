import torch.nn as nn

import torch

# Create sample input
B = 1
L = 2
D = 4
q = torch.arange(8, dtype=torch.float).reshape(B, L, D)
k = torch.arange(8, dtype=torch.float).reshape(B, L, D)
v = torch.arange(8, dtype=torch.float).reshape(B, L, D)

# Use mask of shape (B, L, L)
# Mask out second source element for both elements in target
attn_mask = torch.tensor([[[True, False],
                           [True, False]]])
res1 = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)

# Change masked out element to a very large number - the result doesn't change
k[0, 1, 0] = 1e10
res2 = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
# Result didn't change
print(f"{res1 - res2 = }")

# Change masked out element to NaN - the result becomes NaN
k[0, 1, 0] = torch.nan
res3 = torch.nn.functional.scaled_dot_product_attention(q, k, v, attn_mask=attn_mask)
# Result became NaN
print(f"{res1 - res3 = }")