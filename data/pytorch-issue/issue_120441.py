import torch
from flash_attn import flash_attn_with_kvcache

with torch.device("cuda"):
    q = torch.randn((1, 2, 2, 4), dtype=torch.bfloat16)
    k_cache = torch.randn((1, 5, 2, 4), dtype=torch.bfloat16)
    v_cache = torch.randn((1, 5, 2, 4), dtype=torch.bfloat16)
    k = torch.randn((1, 1, 2, 4), dtype=torch.bfloat16)
    v = torch.randn((1, 1, 2, 4), dtype=torch.bfloat16)
    cache_seqlens = torch.tensor([3], dtype=torch.int32)

torch.library.define(
    "mylib::custom_func",
    "(Tensor q, Tensor(a!) k_cache, Tensor(a!) v_cache, Tensor k, Tensor v, Tensor cache_seqlens) -> Tensor",
)

@torch.library.impl("mylib::custom_func", "cuda")
def custom_func(q, k_cache, v_cache, k, v, cache_seqlens):
    return flash_attn_with_kvcache(
        q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens
    )

@torch.library.impl_abstract("mylib::custom_func")
def custom_func_abstract(q, k_cache, v_cache, k, v, cache_seqlens):
    return torch.empty_like(q)

assert torch.allclose(
    flash_attn_with_kvcache(q, k_cache, v_cache, k=k, v=v, cache_seqlens=cache_seqlens),
    torch.ops.mylib.custom_func(q, k_cache, v_cache, k, v, cache_seqlens),
)
torch.compile(torch.ops.mylib.custom_func, fullgraph=True)(
    q, k_cache, v_cache, k, v, cache_seqlens
)