import torch

# TODO: model memory format on TensorMeta
def _clone_meta(
    a: TensorLikeType, *, memory_format: torch.memory_format
) -> TensorLikeType:
    return TensorMeta(a)