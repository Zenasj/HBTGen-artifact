# jagged.py
import torch
from torch.nested._internal.ops import check_ragged_dim_same, normalize_function, register_jagged_func

@register_jagged_func(
    torch.ops.aten.embedding_dense_backward.default,
    "self: jt, grad_output: jt, num_weights: any, padding_idx: any, scale_grad_by_freq: any",
)
def embedding_dense_backward(func, *args, **kwargs):
    _, new_kwargs = normalize_function(  # type: ignore[misc]
        func, args=args, kwargs=kwargs, normalize_to_only_use_kwargs=True
    )

    indices: NestedTensor = new_kwargs.pop("indices")
    num_weights: int = new_kwargs.pop("num_weights")
    grad_output: NestedTensor = new_kwargs.pop("grad_output")
    if new_kwargs["padding_idx"] != -1 or new_kwargs["scale_grad_by_freq"]:
        raise NotImplementedError("Haven't done this yet")

    check_ragged_dim_same(func, indices, "self", grad_output, "grad_output")
    out = torch.zeros(num_weights, grad_output.size(-1))
    src = grad_output._values
    indices = indices._values.long().unsqueeze(1).expand(-1, src.size(1))
    out.scatter_add_(dim=0, index=indices, src=src)
    return out