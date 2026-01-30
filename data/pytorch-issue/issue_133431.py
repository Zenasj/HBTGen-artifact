import torch.nn as nn

import torch

from torch.nn.attention.flex_attention import flex_attention, create_block_mask

import functools

if __name__ == "__main__":

    B, H, SEQ_LEN, HEAD_DIM = 1, 16, 40320, 16 
    WINDOW_SIZE = 512

    def make_tensor():
        return torch.ones(B, H, SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float16, requires_grad=True)

    q, k, v = make_tensor(), make_tensor(), make_tensor()
    gradOut = torch.ones(B, H, SEQ_LEN, HEAD_DIM, device="cuda", dtype=torch.float16)

    def sliding_window(b, h, q_idx, kv_idx):
            return torch.abs(q_idx - kv_idx) <= WINDOW_SIZE

    block_mask = create_block_mask(
        sliding_window, B=None, H=None, Q_LEN=SEQ_LEN, KV_LEN=SEQ_LEN, _compile=True
    )
    
    attention = functools.partial(flex_attention, block_mask=block_mask) #cache the block mask so its not remade

   
    attention = torch.compile(attention, fullgraph=False)

    out = attention(q, k, v, block_mask=block_mask)
    print(f"Shape of output tensor: {list(out.shape)}")
    out.backward(gradOut, retain_graph=True)
    print(f"Shape of output tensor after bw: {list(out.shape)}")

import os
os.environ['TORCHINDUCTOR_COMPILE_THREADS'] = '1'

import torch
from torch.nn.attention.flex_attention import flex_attention

if __name__ == "__main__":

    torch.set_default_device("cuda")
    torch.manual_seed(0)
    torch._dynamo.config.cache_size_limit = 1000

    B = 16
    H = 8
    S = 1024
    D = 64
    data_type = torch.float16 # <-------- float32 works
    device = "cuda"

    qkv = [
        torch.randn(B, H, S, D, device=device, dtype=data_type, requires_grad=True)
        for _ in range(3)
    ]

    flex_attention_c = torch.compile(flex_attention, dynamic=False)
    out = flex_attention_c(*qkv)