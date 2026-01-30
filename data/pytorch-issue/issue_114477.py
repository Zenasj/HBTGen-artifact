import torch.nn as nn

import torch

from torch.sparse.semi_structured import (
    SparseSemiStructuredTensor,
    to_sparse_semi_structured,
)

@torch.compile(backend="inductor", fullgraph=True)
def my_linear(input, weight):
    return torch.nn.functional.linear(input, weight)

SparseSemiStructuredTensor._FORCE_CUTLASS = True

m, n, k = 1, 32, 64
dtype = torch.half
device = "cuda"

torch.manual_seed(0)

input = torch.rand((m, k), dtype=dtype, device=device)
weight = torch.rand((n, k), dtype=dtype, device=device)

mask = torch.Tensor([1, 0, 0, 1]).to(dtype).to(device).tile((n, k // 4))

dense_weight = weight * mask
dense_result = torch.nn.functional.linear(input, dense_weight)

sparse_weight = to_sparse_semi_structured(dense_weight)
sparse_result = my_linear(input, sparse_weight)

assert torch.allclose(dense_result, sparse_result, rtol=1e-3, atol=1e-3)

@register_meta(aten._sparse_semi_structured_linear)
def meta_sparse_structured_linear(
    input: Tensor,
    weight: Tensor,
    _meta: Tensor,
    bias: Optional[Tensor] = None,
    _activation_opt: Optional[str] = None,
):
    output_sizes = list(input.shape)
    if bias is not None:
        assert weight.size(0) == bias.size(0), "output size mismatch"
    assert weight.size(1) == input.size(-1) / 2
    output_sizes[-1] = weight.size(0)

    transposed_strides = input.new_empty(output_sizes).transpose(-1, -2).stride()

    return input.new_empty(
        output_sizes,
        dtype=input.dtype if input.dtype != torch.int8 else torch.int32,
    ).as_strided(output_sizes, transposed_strides)