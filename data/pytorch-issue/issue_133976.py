import pickle
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

import time


def attention_pytorch_flash(qkv, dropout_p=0.0, causal=True):
    """
    Arguments:
        qkv: (batch_size, seqlen, 3, nheads, head_dim)
        dropout_p: float
    Output:
        output: (batch_size, seqlen, nheads, head_dim)
    """
    # batch_size, seqlen, _, nheads, d = qkv.shape
    q, k, v = qkv.unbind(dim=2)

    q =  q.contiguous()
    v = v.contiguous()
    k = k.contiguous()
    # with torch.backends.cuda.sdp_kernel(enable_flash=False, enable_math=False, enable_mem_efficient=True):
    output = F.scaled_dot_product_attention(q, k, v)
    return output.to(dtype=qkv.dtype)

qkv = torch.randn(1, 65536, 3, 8, 128).cuda().bfloat16()
for _ in range(100):
    output = attention_pytorch_flash(qkv)

torch.cuda.synchronize()
start = time.time()
for _ in range(100):
    output = attention_pytorch_flash(qkv)
torch.cuda.synchronize()
print("attention_pytorch_flash Time: ", (time.time()-start) / 100)