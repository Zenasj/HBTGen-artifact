import torch.nn as nn

import torch
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

flex_attention = torch.compile(flex_attention, dynamic=False)


qk_dims = 64
v_dims = 128
q_heads = 8
kv_heads = 4

seq_len = 10000


class Rig(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.k = torch.randn((1, kv_heads, seq_len, qk_dims), requires_grad=True, device='cuda', dtype=torch.float)
        self.v = torch.randn((1, kv_heads, seq_len, v_dims), requires_grad=True, device='cuda', dtype=torch.float)
        self.q = torch.randn((1, q_heads, seq_len, qk_dims), requires_grad=True, device='cuda', dtype=torch.float)


class FlexRig(Rig):
    def forward(self):
        u = flex_attention(
            self.q, self.k, self.v, 
            enable_gqa=True, 
        )
        print(self.q.shape, self.k.shape, self.v.shape, '->', u.shape)

print('Flex without compile: ', end='')
FlexRig()()

print('Flex with compile: ', end='')
torch.compile(FlexRig())()

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

flex_attention_compiled = torch.compile(flex_attention, dynamic=False, fullgraph=True)


qk_dims = 64
v_dims = 128
q_heads = 8
kv_heads = 4

seq_len = 10000


class Rig(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.k = torch.randn(
            (1, kv_heads, seq_len, qk_dims),
            requires_grad=True,
            device="cuda",
            dtype=torch.float,
        )
        self.v = torch.randn(
            (1, kv_heads, seq_len, v_dims),
            requires_grad=True,
            device="cuda",
            dtype=torch.float,
        )
        self.q = torch.randn(
            (1, q_heads, seq_len, qk_dims),
            requires_grad=True,
            device="cuda",
            dtype=torch.float,
        )


class FlexRig(Rig):
    def forward(self):
        u = flex_attention(
            self.q,
            self.k,
            self.v,
            enable_gqa=True,
        )
        return u
        print(self.q.shape, self.k.shape, self.v.shape, "->", u.shape)


print("Flex without compile: ", end="")
FlexRig()()


print("Flex with compile: ", end="")
out = torch.compile(FlexRig(), fullgraph=True)()
print(out.shape)

import torch
from torch.nn.attention.flex_attention import create_block_mask, flex_attention

flex_attention_compiled = torch.compile(flex_attention, dynamic=False, fullgraph=True)


qk_dims = 64
v_dims = 128
q_heads = 8
kv_heads = 4

seq_len = 10000


class Rig(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.k = torch.randn(
            (1, kv_heads, seq_len, qk_dims),
            requires_grad=True,
            device="cuda",
            dtype=torch.float,
        )
        self.v = torch.randn(
            (1, kv_heads, seq_len, v_dims),
            requires_grad=True,
            device="cuda",
            dtype=torch.float,
        )
        self.q = torch.randn(
            (1, q_heads, seq_len, qk_dims),
            requires_grad=True,
            device="cuda",
            dtype=torch.float,
        )


class FlexRig(Rig):
    def forward(self):
        u = flex_attention(
            self.q,
            self.k,
            self.v,
            enable_gqa=True,
        )
        print(self.q.shape, self.k.shape, self.v.shape, "->", u.shape)
        return u


print("Flex without compile: ", end="")
FlexRig()()


print("Flex with compile: ", end="")
out = torch.compile(FlexRig(), fullgraph=False)()
print(out.shape)