# torch.rand(16, 1, 8, 128, dtype=torch.bfloat16)  # First tensor in input tuple (q)

import torch
from torch import nn

# Define the custom flash attention operation
torch.library.define(
    "mylib::flash_attn_with_kvcache_wrapper",
    "(Tensor q, Tensor(a!) k_cache, Tensor(b!) v_cache, Tensor k, Tensor v, Tensor cache_seqlens, Tensor block_table) -> Tensor",
)

@torch.library.impl("mylib::flash_attn_with_kvcache_wrapper", "cuda")
def flash_attn_with_kvcache_wrapper(
    q,
    k_cache,
    v_cache,
    k,
    v,
    cache_seqlens,
    block_table,
):
    # Placeholder implementation; actual implementation requires flash_attn library
    return q  # Dummy output for structure

@torch.library.impl_abstract("mylib::flash_attn_with_kvcache_wrapper")
def flash_attn_with_kvcache_wrapper_abstract(
    q,
    k_cache,
    v_cache,
    k,
    v,
    cache_seqlens,
    block_table,
):
    return torch.empty_like(q)

class MyModel(nn.Module):
    def forward(self, inputs):
        q, k_cache, v_cache, k, v, cache_seqlens, block_table = inputs
        return torch.ops.mylib.flash_attn_with_kvcache_wrapper(
            q, k_cache, v_cache, k, v, cache_seqlens, block_table
        )

def my_model_function():
    return MyModel()

def GetInput():
    n_blocks = 20_000
    n_batch = 16
    n_head = 8
    d_head = 128
    q_len = 1

    q = torch.randn(n_batch, q_len, n_head, d_head, device="cuda", dtype=torch.bfloat16)
    k_cache = torch.randn(n_blocks, 256, n_head, d_head, device="cuda", dtype=torch.bfloat16)
    v_cache = torch.rand_like(k_cache)
    k = torch.randn(n_batch, 1, n_head, d_head, device="cuda", dtype=torch.bfloat16)
    v = torch.rand_like(k)
    cache_seqlens = torch.tensor([2500] * n_batch, device="cuda", dtype=torch.int32)
    block_table = torch.arange(10, device="cuda", dtype=torch.int32).expand(n_batch, -1)

    return (q, k_cache, v_cache, k, v, cache_seqlens, block_table)

