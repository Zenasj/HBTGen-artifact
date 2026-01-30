import torch

_length: List[int] = (
    _length_per_key_from_stride_per_key(torch.diff(offsets), stride_per_key)
    if variable_stride_per_key
    else torch.sum(torch.diff(offsets).view(-1, stride), dim=1).tolist()
)