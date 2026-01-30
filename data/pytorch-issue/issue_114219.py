import torch.nn as nn

import torch
def fn(input):
    input = input.unsqueeze(0)
    kernel_size = 2
    stride = 2
    padding = 1
    dilation = 1
    return_indices = True
    ceil_mode = True
    fn_res = torch.nn.functional.max_pool1d(
        input, kernel_size, stride=stride, padding=padding,
        dilation=dilation, return_indices=return_indices, ceil_mode=ceil_mode
    )   
    return fn_res
input_tensor = torch.tensor([1, 1, 1, 8], dtype=torch.float64, requires_grad=True)
torch.autograd.gradcheck(fn, input_tensor)