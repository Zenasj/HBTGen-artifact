import torch
import torch.nn as nn

m = nn.MultiheadAttention(embed_dim=4, num_heads=4, batch_first=True)
N, L, E = 2, 8, 4
x = torch.randn(N, L, E)
key_padding_mask = torch.randn(N, L).bool()
attn_mask = torch.randn(N, L, L).repeat(m.num_heads, 1, 1)

m.train()
ao, aow = m(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask)
print(ao.shape)

m.eval()
with torch.no_grad():
    # Modify attn_mask to match expected shape
    attn_mask_eval = attn_mask.view(-1, L, L)
    try:
        ao, aow = m(x, x, x, key_padding_mask=key_padding_mask, attn_mask=attn_mask_eval)
        print(ao.shape)
    except:
        print("Error occurred during evaluation")