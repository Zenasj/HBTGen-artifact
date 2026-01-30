import torch.nn as nn

QL = query.size(2)
KL = key.size(2)
def causal_mask(b, h, q_idx, kv_idx):
    return q_idx - QL >= kv_idx - KL

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

def causal_attention(
    query,
    key,
    value,
):
    # all shapes  Bs x Nh x Len x Dim
    B = query.size(0)
    H = query.size(1)
    QL = query.size(2)
    KL = key.size(2)

    def causal_mask(b, h, q_idx, kv_idx):
        return q_idx - QL >= kv_idx - KL

    block_mask = create_block_mask(causal_mask, B, H, QL, KL, device=query.device)
    return flex_attention(
        query,
        key,
        value,
        None,
        block_mask,
    )


def test(ql, kl):
    bs = 32
    nh = 8
    hd = 64
    q = torch.rand(
        bs, nh, ql, hd, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    k = torch.rand(
        bs, nh, kl, hd, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    v = torch.rand(
        bs, nh, kl, hd, dtype=torch.bfloat16, device="cuda", requires_grad=True
    )
    causal_attention(q, k, v)
    print(f"test({ql}, {kl}) worked")


print("torch.__version__", torch.__version__)

# First calls always succeed.
test(512, 512)
test(512, 512)
# These calls fail, unless the above are commented out. 
test(512, 1024)
test(512, 1024)
test(512, 512)