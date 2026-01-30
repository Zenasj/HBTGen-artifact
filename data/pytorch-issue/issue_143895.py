import torch
import numpy as np

def _kernel_make_viewless_tensor(inp, requires_grad):
    """Make a viewless tensor.

    View tensors have the undesirable side-affect of retaining a reference
    to the originally-viewed tensor, even after manually setting the '.data'
    field. This method creates a new tensor that links to the old tensor's
    data, without linking the viewed tensor, referenced via the '._base'
    field.
    """
    # out = torch.empty((1,), dtype=inp.dtype, device=inp.device, requires_grad=requires_grad)
    # out.data = inp.data
    # return out
    return inp.detach().requires_grad_(requires_grad)