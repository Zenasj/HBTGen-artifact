import torch.nn as nn

py
import torch
def fn(tensor):
    fn_res = torch.nn.init.constant_(tensor, 0.0)
    return fn_res
tensor = torch.tensor([1., 1.], dtype=torch.float64, requires_grad=True)

torch.autograd.gradcheck(fn, (tensor,))