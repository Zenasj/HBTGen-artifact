import torch

@torch.compile
def fn(a, lengths):
    # deliberately force a reinterpret_tensor call on lengths
    size = list(lengths.size())
    size[0] //= 2
    stride = list(lengths.stride())
    stride[0] *= 2
    strided_lengths = lengths.as_strided(size, stride)
    return torch.ops.aten.segment_reduce(a, "max", axis=0, unsafe=False, initial=1, lengths=strided_lengths)
    
data = torch.rand(5)
lengths = torch.tensor([0, 0, 1, 1, 2, 2, 2, 2], dtype=torch.long)

fn(data, lengths)