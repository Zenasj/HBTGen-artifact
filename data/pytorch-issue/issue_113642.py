import torch.nn as nn

import torch
import numpy as np

grad_output_ncthw = torch.tensor([[[[[1]]], [[[1]]]]], dtype = torch.float32).cuda()
temp = torch.tensor([[[[[1, 1]]]]], dtype = torch.float32).cuda()
grad_output_permute2ncthw = temp.permute(0, 4, 1, 2, 3).contiguous()
print("equal:", grad_output_ncthw == grad_output_permute2ncthw)
print("1st:", grad_output_ncthw.shape, grad_output_ncthw.stride(), grad_output_ncthw.is_contiguous())
print("2nd:", grad_output_permute2ncthw.shape, grad_output_permute2ncthw.stride(), grad_output_permute2ncthw.is_contiguous())

use_1st = False
if use_1st:
    print("use 1st")
    grad_output = grad_output_ncthw
else:
    print("use 2nd")
    grad_output = grad_output_permute2ncthw

interp_input = torch.randn([1, 2, 4, 8, 16], dtype = torch.float32).cuda()

interp_input.requires_grad = True

interp_output = torch.nn.functional.interpolate(interp_input, scale_factor = [0.25, 0.125, 0.0625], mode = "trilinear", align_corners = False, recompute_scale_factor = False)

interp_output.backward(grad_output)
torch.set_printoptions(threshold = 200000)
grad_x = interp_input.grad
print(grad_x.shape, grad_x.stride())
print("grad_input", grad_x)