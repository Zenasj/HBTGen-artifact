import torch.nn as nn

import torch
import torch.nn.functional as F

head_num, seq_len, embed_dim = 16, 16, 80
bsz = 1

q = torch.randn(head_num, seq_len, embed_dim)
k = torch.randn(head_num, seq_len, embed_dim)
v = torch.randn(head_num, seq_len, embed_dim)
attention_mask = torch.ones(1, seq_len, seq_len)

oo_cpu = F.scaled_dot_product_attention(
    q.to("cpu"),
    k.to("cpu"),
    v.to("cpu"),
    attention_mask.to("cpu"),
    dropout_p=0.0
)

if torch.backends.mps.is_available():
    oo_mps = F.scaled_dot_product_attention(
        q.to("mps"),
        k.to("mps"),
        v.to("mps"),
        attention_mask.to("mps"),
        dropout_p=0.0
    )
    assert torch.allclose(oo_cpu, oo_mps.to("cpu"), atol=1e-5)