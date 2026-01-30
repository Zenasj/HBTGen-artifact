# %%
import os
from flash_attn import flash_attn_with_kvcache
import torch
import torch.utils.benchmark as benchmark

# Set the CUDA device
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Define the custom flash attention operation
torch.library.define(
    "mylib::flash_attn_with_kvcache_wrapper",
    "(Tensor q, Tensor(a!) k_cache, Tensor(a!) v_cache, Tensor k, Tensor v, Tensor cache_seqlens, Tensor block_table) -> Tensor",
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
    return flash_attn_with_kvcache(
        q,
        k_cache,
        v_cache,
        k=k,
        v=v,
        cache_seqlens=cache_seqlens,
        block_table=block_table,
        causal=True,
    )


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


# Initialize tensors
n_blocks = 20_000
n_batch = 16
n_head = 8
d_head = 128
q_len = 1

q = torch.randn(n_batch, q_len, n_head, d_head, device="cuda", dtype=torch.bfloat16)
k_cache = torch.randn(
    n_blocks, 256, n_head, d_head, device="cuda", dtype=torch.bfloat16
)
v_cache = torch.rand_like(k_cache)
k = torch.randn(n_batch, 1, n_head, d_head, device="cuda", dtype=torch.bfloat16)
v = torch.rand_like(k)
cache_seqlens = torch.tensor([2500] * n_batch, device="cuda", dtype=torch.int32)
block_table = torch.arange(10, device="cuda", dtype=torch.int32).expand(n_batch, -1)

# Run the custom flash attention operation
torch.ops.mylib.flash_attn_with_kvcache_wrapper(
    q, k_cache, v_cache, k, v, cache_seqlens, block_table
)


# Function to benchmark the custom flash attention operation
def run_flash_attn_with_kvcache_wrapper():
    x = 0
    for _ in range(100):
        x = x + torch.ops.mylib.flash_attn_with_kvcache_wrapper(
            q, k_cache, v_cache, None, None, cache_seqlens, block_table
        )
    return x


compiled_flash_attn_with_kvcache_wrapper = torch.compile(
    run_flash_attn_with_kvcache_wrapper
)


# Benchmark the custom flash attention operation
print(
    "Uncompiled: ",
    benchmark.Timer(
        stmt="run_flash_attn_with_kvcache_wrapper()",
        globals={
            "run_flash_attn_with_kvcache_wrapper": run_flash_attn_with_kvcache_wrapper
        },
    ).timeit(100),
)

# With compilation
print(
    "Compiled: ",
    benchmark.Timer(
        stmt="compiled_flash_attn_with_kvcache_wrapper()",
        globals={
            "compiled_flash_attn_with_kvcache_wrapper": compiled_flash_attn_with_kvcache_wrapper
        },
    ).timeit(100),
)