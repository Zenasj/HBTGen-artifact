import torch
from torch.nested._internal.nested_tensor import nested_view_from_values_offsets

def fn(values, offsets):
    return nested_view_from_values_offsets(values, offsets, max_seqlen=200, min_seqlen=1)

def get_values_offsets(batch_size=16, max_seqlen=200, inner_dim=64):
    lengths = torch.randint(1, max_seqlen, (batch_size,), dtype=torch.int32)
    offsets = torch.zeros((batch_size + 1,), dtype=torch.int32)
    torch.cumsum(lengths, dim=0, out=offsets[1:])
    values = torch.randn((offsets[-1].item(), inner_dim), dtype=torch.bfloat16, requires_grad=True)
    return values, offsets

values, offsets = get_values_offsets()
torch.compile(fn)(values, offsets)