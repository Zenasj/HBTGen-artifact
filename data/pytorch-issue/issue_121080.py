import torch

def torch_expand_indptr(indptr, dtype, nodes=None, output_size=None):
    if nodes is None:
        nodes = torch.arange(len(indptr) - 1, dtype=dtype, device=indptr.device)
    return nodes.to(dtype).repeat_interleave(indptr.diff(), output_size=output_size)