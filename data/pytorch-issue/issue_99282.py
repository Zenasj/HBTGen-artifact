import torch 
import torch.nn as nn

batch_size = 4
seq_len = 3
embedding_dim = 8
num_heads = 2

mha=nn.MultiheadAttention(num_heads=num_heads, embed_dim=embedding_dim, batch_first=True)
x = torch.randn(batch_size, seq_len, embedding_dim)

mask = nn.Transformer.generate_square_subsequent_mask(seq_len)

with torch.backends.cuda.sdp_kernel(enable_math=True, enable_mem_efficient=False, enable_flash=False):
    no_mask = mha(x,x,x, need_weights=False)[0]
    with_attn_mask = mha(x,x,x, need_weights=True, attn_mask=mask)[0]
    with_is_causal_need_weights = mha(x,x,x, need_weights=True, is_causal=True)[0]
    with_is_causal_no_need_weights = mha(x,x,x, need_weights=False, is_causal=True)[0]
    
#succeeds    
assert with_attn_mask.allclose(with_is_causal_no_need_weights)

#both should succeed but fail
assert with_attn_mask.allclose(with_is_causal_need_weights), "is_causal should match regardless of 'need_weights'"
assert not no_mask.allclose(with_is_causal_need_weights),  "no mask should NOT match is_causal=True, need_weights=True"