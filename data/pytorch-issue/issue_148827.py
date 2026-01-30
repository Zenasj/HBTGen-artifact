class Attention(nn.Module):
    def __init__(
        self,
        q_ch: int,
        kv_ch: Optional[int] = None,
        qk_embed_dim: Optional[int] = None,
        v_embed_dim: Optional[int] = None,
        linear_bias: bool = False,
        num_heads: int = 1,
    ):
        self.q_ch = q_ch
        self.kv_ch = kv_ch or self.q_ch
        self.qk_embed_dim = qk_embed_dim or self.q_ch
        self.v_embed_dim = v_embed_dim or self.kv_ch
        self.num_heads = num_heads
        assert (
            not self.qk_embed_dim % num_heads and not self.v_embed_dim % num_heads
        ), "The dimension of the embeddings in Attention must be divisible by the number of heads."
        super().__init__()

        self.q_proj = nn.Linear(self.q_ch, self.qk_embed_dim, bias=linear_bias)
        self.kv_proj = nn.Linear(
            self.kv_ch, self.qk_embed_dim + self.v_embed_dim, bias=linear_bias
        )
        self.o_proj = nn.Linear(self.v_embed_dim, self.q_ch, bias=linear_bias)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: torch.nn.attention.flex_attention.BlockMask,
    ) -> torch.Tensor:
        return torch.nn.attention.flex_attention.flex_attention(
            q, k, v, block_mask=block_mask
        )

    def forward(
        self, x: torch.Tensor, block_mask: torch.nn.attention.flex_attention.BlockMask
    ) -> torch.Tensor:
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        k = kv[..., : self.qk_embed_dim]
        v = kv[..., self.qk_embed_dim :]
        q = q.reshape((q.shape[0], q.shape[1], self.num_heads, -1)).transpose(1, 2)
        k = k.reshape((k.shape[0], k.shape[1], self.num_heads, -1)).transpose(1, 2)
        v = v.reshape((v.shape[0], v.shape[1], self.num_heads, -1)).transpose(1, 2)
        return self.o_proj(
            self.scaled_dot_product_attention(q, k, v, block_mask)
            .transpose(1, 2)
           .reshape((x.shape[0], x.shape[1], -1))
        )

def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: torch.nn.attention.flex_attention.BlockMask,
    ) -> torch.Tensor:
        return torch.nn.attention.flex_attention.flex_attention(
            q.contiguous(), k.contiguous(), v.contiguous(), block_mask=block_mask
        )

def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: torch.nn.attention.flex_attention.BlockMask,
    ) -> torch.Tensor:
        print("", end="")
        return torch.nn.attention.flex_attention.flex_attention(
            q, k, v, block_mask=block_mask
        )

from typing import Any, Dict, Optional, Tuple


import torch
import torch.nn as nn
import torch.nn.attention.flex_attention
import torch.nn.functional as F


class Attention(nn.Module):
    def __init__(
        self,
        q_ch: int,
        kv_ch: Optional[int] = None,
        qk_embed_dim: Optional[int] = None,
        v_embed_dim: Optional[int] = None,
        linear_bias: bool = False,
        num_heads: int = 1,
    ):
        self.q_ch = q_ch
        self.kv_ch = kv_ch or self.q_ch
        self.qk_embed_dim = qk_embed_dim or self.q_ch
        self.v_embed_dim = v_embed_dim or self.kv_ch
        self.num_heads = num_heads
        assert (
            not self.qk_embed_dim % num_heads and not self.v_embed_dim % num_heads
        ), "The dimension of the embeddings in Attention must be divisible by the number of heads."
        super().__init__()

        self.q_proj = nn.Linear(self.q_ch, self.qk_embed_dim, bias=linear_bias)
        self.kv_proj = nn.Linear(
            self.kv_ch, self.qk_embed_dim + self.v_embed_dim, bias=linear_bias
        )
        self.o_proj = nn.Linear(self.v_embed_dim, self.q_ch, bias=linear_bias)

    def scaled_dot_product_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        block_mask: torch.nn.attention.flex_attention.BlockMask,
    ) -> torch.Tensor:
        return torch.nn.attention.flex_attention.flex_attention(
            q, k, v, block_mask=block_mask
        )

    def forward(
        self, x: torch.Tensor, block_mask: torch.nn.attention.flex_attention.BlockMask
    ) -> torch.Tensor:
        q = self.q_proj(x)
        kv = self.kv_proj(x)
        k = kv[..., : self.qk_embed_dim]
        v = kv[..., self.qk_embed_dim :]
        q = q.reshape((q.shape[0], q.shape[1], self.num_heads, -1)).transpose(1, 2)
        k = k.reshape((k.shape[0], k.shape[1], self.num_heads, -1)).transpose(1, 2)
        v = v.reshape((v.shape[0], v.shape[1], self.num_heads, -1)).transpose(1, 2)
        return self.o_proj(
            self.scaled_dot_product_attention(q, k, v, block_mask)
            .transpose(1, 2)
            .reshape((x.shape[0], x.shape[1], -1))
        )


def create_block_mask(b: int, h: int, q_seq_len: int, k_seq_len: int):
    def mask(b, h, q_idx, k_idx):
        return q_idx >= k_idx

    return torch.nn.attention.flex_attention.create_block_mask(
        mask, b, h, q_seq_len, k_seq_len
    )


@torch.compile
class TestModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.attn_layer = Attention(768, num_heads=24)
        self.convs = nn.Sequential(
            *(nn.Conv2d(768, 768, 3, padding=1) for _ in range(3))
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        block_mask = create_block_mask(8, 24, 256, 256)
        y = self.convs(x).view((x.shape[0], x.shape[1], -1)).transpose(1, 2)
        o = self.attn_layer(y, block_mask)
        return self.convs(o.transpose(1, 2).reshape(x.shape))


if __name__ == "__main__":
    test_model = TestModel().cuda()
    x = torch.randn((8, 768, 16, 16), device="cuda", requires_grad=True)
    z = test_model(x)
    z.sum().backward()
    torch.cuda.synchronize()