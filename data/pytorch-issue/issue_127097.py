import torch

values = torch.randn((18, 16))
offsets = torch.tensor([0, 2, 3, 6, 15, 18])
like_values = torch.randn_like(values)

# this marks values as dynamic
nt = torch.nested.nested_tensor_from_jagged(values, offsets)

def fn(values, same_size):
    return values + same_size

torch.compile(fn)(values, like_values)